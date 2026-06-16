"""
Dynamic context loading hook module.

Loads capability context on demand rather than statically at session start.
A classifier inspects each user message for trigger keywords and activates
matching context blocks. The loader hook then injects those blocks ephemerally
before each LLM call — keeping the system prompt lean until topics are needed.

Architecture:
  - DynamicContextManifest: parses a YAML manifest declaring all available blocks
  - LazyContextBlock:       one block (name, description, path, triggers, cached content)
  - ContextClassifier:      keyword matcher; accumulates active_tags across the session
  - ContextLoaderHook:      provider:request handler — classifies + injects content
  - mount():                wires everything together and registers the hook

Manifest YAML format:
    context_blocks:
      - name: imessage-knowledge
        description: "iMessage database queries, AppleScript send/receive, contact resolution"
        path: "/absolute/path/to/imessage-knowledge.md"
        triggers: ["message", "text", "imessage", "sms", "texts", "texted"]

      - name: calendar-knowledge
        description: "Google Calendar API, event creation, recurring events, attendees"
        path: "/absolute/path/to/calendar-knowledge.md"
        triggers: ["calendar", "event", "schedule", "meeting", "appointment"]

Config keys (in bundle YAML under module config):
    manifest_path: "/path/to/manifest.yaml"   # required; absolute or relative to working_dir
    priority:      20                          # hook priority (default: 20)
    max_file_bytes: 65536                      # per-file size cap in bytes (default: 64KB)
    max_injection_tokens: 32768               # total injection budget per turn (default: ~32K tokens)
"""

from __future__ import annotations

# Amplifier module metadata
__amplifier_module_type__ = "hook"

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from amplifier_core import HookResult  # type: ignore[import-untyped]
from amplifier_core import ModuleCoordinator  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LazyContextBlock:
    """
    A single capability context block declared in the manifest.

    Content is loaded from disk on first access and cached in memory.
    The external cache dict (keyed by path) is the primary store so that
    the same file is never read twice even across multiple block instances
    that happen to point to the same path.
    """

    name: str
    description: str
    path: str
    triggers: list[str] = field(default_factory=list)
    # Per-instance content cache (set after first successful read)
    _content: str | None = field(default=None, repr=False, init=False, compare=False)

    def load(self, cache: dict[str, str], max_bytes: int = 0) -> str | None:
        """
        Return block content, reading from disk the first time.

        Uses *cache* (path -> content) as the shared in-memory store so
        files are only read once per session even if multiple callers ask
        for the same block.  Also sets ``_content`` on the instance as a
        secondary fast path.

        Args:
            cache:     Shared path -> content mapping (mutated in-place).
            max_bytes: Per-file size cap in bytes.  When > 0 and the file
                       exceeds this limit, a degradation notice is cached
                       and returned instead of the full content so the
                       model knows the file exists and how to access it.
        """
        # Fast path: already loaded on this instance
        if self._content is not None:
            return self._content

        # Shared cache hit
        if self.path in cache:
            self._content = cache[self.path]
            return self._content

        # Read from disk
        try:
            content = Path(self.path).expanduser().read_text(encoding="utf-8")
            content_bytes = len(content.encode("utf-8"))
            if max_bytes > 0 and content_bytes > max_bytes:
                estimated_tokens = content_bytes // 4
                notice = (
                    f"[Content too large: {self.name} at {self.path} "
                    f"is {content_bytes:,} bytes (~{estimated_tokens:,} tokens), "
                    f"exceeds limit of {max_bytes:,} bytes. "
                    f"Use read_file to access specific sections.]"
                )
                logger.warning(
                    "dynamic-context: '%s' at %s is %d bytes, exceeds limit of %d — returning degradation notice",
                    self.name,
                    self.path,
                    content_bytes,
                    max_bytes,
                )
                # Cache the notice (not the full content) so we don't re-read from disk
                cache[self.path] = notice
                self._content = notice
                return notice
            cache[self.path] = content
            self._content = content
            logger.debug(
                "dynamic-context: Loaded '%s' from %s (%d chars)",
                self.name,
                self.path,
                len(content),
            )
            return content
        except OSError as exc:
            logger.warning(
                "dynamic-context: Cannot read '%s' at %s: %s", self.name, self.path, exc
            )
            return None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class DynamicContextManifest:
    """
    Parsed manifest of all declared capability context blocks.

    Reads the YAML file once at mount time.  Provides lookup by name
    and iteration over all blocks.

    Expected YAML shape::

        context_blocks:
          - name: imessage-knowledge
            description: "iMessage queries, AppleScript, contact resolution"
            path: "/absolute/path/to/file.md"
            triggers: ["message", "text", "imessage", "sms"]
    """

    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = manifest_path
        self.blocks: list[LazyContextBlock] = []
        self._load()

    def _load(self) -> None:
        """Parse the YAML manifest and populate ``self.blocks``."""
        try:
            raw = Path(self.manifest_path).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "dynamic-context: Cannot read manifest at %s: %s",
                self.manifest_path,
                exc,
            )
            return

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            logger.warning(
                "dynamic-context: Invalid YAML in manifest %s: %s",
                self.manifest_path,
                exc,
            )
            return

        if not data or not isinstance(data, dict):
            logger.warning(
                "dynamic-context: Manifest %s is empty or not a mapping",
                self.manifest_path,
            )
            return

        raw_blocks = data.get("context_blocks")
        if not raw_blocks or not isinstance(raw_blocks, list):
            logger.warning(
                "dynamic-context: Manifest %s missing 'context_blocks' list",
                self.manifest_path,
            )
            return

        for entry in raw_blocks:
            if not isinstance(entry, dict):
                logger.warning(
                    "dynamic-context: Skipping non-dict block entry: %r", entry
                )
                continue
            try:
                block = LazyContextBlock(
                    name=str(entry["name"]),
                    description=str(entry.get("description", "")),
                    path=str(entry["path"]),
                    triggers=[str(t) for t in entry.get("triggers", [])],
                )
                self.blocks.append(block)
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "dynamic-context: Skipping invalid block entry %r: %s", entry, exc
                )

        logger.info(
            "dynamic-context: Manifest loaded — %d blocks from %s",
            len(self.blocks),
            self.manifest_path,
        )

    def find(self, name: str) -> LazyContextBlock | None:
        """Return the block with the given name, or None."""
        for block in self.blocks:
            if block.name == name:
                return block
        return None

    def all_names(self) -> list[str]:
        """Return a list of all block names in declaration order."""
        return [b.name for b in self.blocks]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ContextClassifier:
    """
    Keyword-based classifier that accumulates active context tags.

    Tags are *additive* — once a block is activated it stays active for
    the rest of the session (no eviction).  This mirrors how human memory
    works during a conversation: once you start talking about iMessage
    you probably need that context for subsequent turns too.

    Matching rules:
      - Case-insensitive throughout.
      - Single-word triggers: whole-word boundary match (\\b).
      - Multi-word / compound triggers: plain substring match.
    """

    def __init__(self, manifest: DynamicContextManifest) -> None:
        self.manifest = manifest
        self.active_tags: set[str] = set()

    def classify(self, message: str) -> set[str]:
        """
        Inspect *message* and activate any blocks whose triggers match.

        Returns the *full* set of currently active tags (including tags
        activated in previous turns).
        """
        if not message:
            return set(self.active_tags)

        message_lower = message.lower()
        newly_matched: set[str] = set()

        for block in self.manifest.blocks:
            if block.name in self.active_tags:
                continue  # Already active — skip expensive regex work

            for trigger in block.triggers:
                trigger_lower = trigger.lower()

                if " " in trigger_lower:
                    # Compound trigger: substring match
                    matched = trigger_lower in message_lower
                else:
                    # Single-word trigger: word-boundary match
                    pattern = r"\b" + re.escape(trigger_lower) + r"\b"
                    matched = bool(re.search(pattern, message_lower))

                if matched:
                    self.active_tags.add(block.name)
                    newly_matched.add(block.name)
                    break  # One trigger match is enough for this block

        if newly_matched:
            logger.info(
                "dynamic-context: Classifier activated new tags %s (total active: %s)",
                sorted(newly_matched),
                sorted(self.active_tags),
            )

        return set(self.active_tags)

    def activate(self, name: str) -> None:
        """Manually activate a block by name (e.g. called by load_context tool)."""
        if name not in self.active_tags:
            self.active_tags.add(name)
            logger.info("dynamic-context: Manually activated tag '%s'", name)


# ---------------------------------------------------------------------------
# Loader hook
# ---------------------------------------------------------------------------


class ContextLoaderHook:
    """
    provider:request hook that injects dynamic context before every LLM call.

    On each invocation:
      1. Extracts the last user message from event data.
      2. Runs the classifier to update ``active_tags``.
      3. Always injects a *thin manifest* listing all available blocks so
         the LLM knows what exists and can call ``load_context`` explicitly.
      4. For each active block, loads and injects the full file content.

    File reads are cached in ``_content_cache`` so each path is read at
    most once per session.
    """

    def __init__(
        self,
        manifest: DynamicContextManifest,
        classifier: ContextClassifier,
        content_cache: dict[str, str],
        priority: int = 20,
        max_file_bytes: int = 65536,
        max_injection_tokens: int = 32768,
    ) -> None:
        self.manifest = manifest
        self.classifier = classifier
        self._content_cache = content_cache
        self.priority = priority
        self.max_file_bytes = max_file_bytes
        self.max_injection_tokens = max_injection_tokens

    def register(self, hooks: Any) -> None:
        """Register on ``provider:request`` (fires right before each LLM call)."""
        hooks.register(
            "provider:request",
            self.on_provider_request,
            priority=self.priority,
            name="hooks-dynamic-context",
        )

    async def on_provider_request(self, event: str, data: dict[str, Any]) -> HookResult:
        """
        Build and inject dynamic context before the LLM call.

        Always injects the thin manifest.  Also injects full content for
        every block currently in ``classifier.active_tags``.
        """
        if not self.manifest.blocks:
            return HookResult(action="continue")

        # Step 1: classify the latest user message
        last_user_message = self._extract_last_user_message(data)
        if last_user_message:
            self.classifier.classify(last_user_message)

        # Step 2: thin manifest — always present so the LLM can call load_context
        thin_manifest = self._build_thin_manifest()

        # Step 3: load content for every active block (budget-aware)
        content_parts: list[str] = []
        tokens_used = len(thin_manifest) // 4  # thin manifest counts against budget
        skipped_blocks: list[str] = []
        for block_name in sorted(self.classifier.active_tags):
            block = self.manifest.find(block_name)
            if block is None:
                logger.warning(
                    "dynamic-context: Active tag '%s' not found in manifest", block_name
                )
                continue
            content = block.load(self._content_cache, max_bytes=self.max_file_bytes)
            if content:
                content_tokens = len(content) // 4
                if (
                    self.max_injection_tokens > 0
                    and tokens_used + content_tokens > self.max_injection_tokens
                ):
                    skipped_blocks.append(block_name)
                    logger.warning(
                        "dynamic-context: Skipping '%s' (%d tokens) — would exceed injection budget (%d/%d)",
                        block_name,
                        content_tokens,
                        tokens_used,
                        self.max_injection_tokens,
                    )
                    continue
                tokens_used += content_tokens
                content_parts.append(
                    f'<context_file source="dynamic-context:{block_name}">\n{content}\n</context_file>'
                )

        if skipped_blocks:
            budget_notice = (
                f"[Budget limit reached: {len(skipped_blocks)} context block(s) skipped "
                f"({', '.join(skipped_blocks)}). "
                f"Use load_context or read_file to access them manually.]"
            )
            content_parts.append(budget_notice)

        # Step 4: assemble injection
        parts = [thin_manifest]
        parts.extend(content_parts)
        context_injection = "\n\n".join(parts)

        return HookResult(
            action="inject_context",
            context_injection=context_injection,
            context_injection_role="user",
            ephemeral=True,  # Not stored in conversation history
            suppress_output=True,  # Silent — don't surface to user
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_last_user_message(self, data: dict[str, Any]) -> str:
        """
        Pull the text of the most recent ``role=user`` message out of
        the provider request event data.

        Handles both dict-style messages and object-style messages, and
        both plain-string content and structured content-block lists.
        """
        messages = data.get("messages", [])
        for message in reversed(messages):
            # Support dict messages (common in tests / wire format)
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")
            else:
                role = getattr(message, "role", "")
                content = getattr(message, "content", "")

            if role != "user":
                continue

            # Plain string content
            if isinstance(content, str):
                return content

            # Structured content blocks
            if isinstance(content, list):
                texts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    elif (
                        hasattr(block, "type")
                        and getattr(block, "type", None) == "text"
                    ):
                        texts.append(getattr(block, "text", ""))
                return " ".join(texts)

        return ""

    def _build_thin_manifest(self) -> str:
        """
        Produce the always-injected thin manifest block.

        Lists every available context block with its description so the
        LLM knows what exists.  Loaded blocks are marked ``[loaded]``.
        """
        lines = [
            '<system-reminder source="dynamic-context-manifest">',
            "Available capability context (loaded automatically when relevant,"
            " or use load_context tool to load manually):",
        ]
        for block in self.manifest.blocks:
            if block.name in self.classifier.active_tags:
                # Check if content is a degradation notice
                if block._content and block._content.startswith("[Content too large:"):
                    marker = " [too large — use read_file]"
                else:
                    marker = " [loaded]"
            else:
                marker = ""
            lines.append(f"- {block.name}: {block.description}{marker}")
        lines.append("</system-reminder>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def mount(
    coordinator: ModuleCoordinator, config: dict[str, Any] | None = None
) -> None:
    """
    Mount the dynamic context loading hook.

    Reads ``manifest_path`` from *config*, loads the manifest, creates the
    classifier and loader, registers the hook, and stores shared state on
    ``coordinator.session_state`` so the companion ``load_context`` tool can
    access the same manifest and classifier.

    Args:
        coordinator: Module coordinator provided by the Amplifier runtime.
        config:      Module configuration dict.  Recognised keys:

            manifest_path (str, required):
                Absolute path, or path relative to the session working
                directory, pointing to the capability manifest YAML.

            priority (int, default 20):
                Hook priority.  Runs after status-context (0) and
                the reminder hook (priority 10) by default.

            max_file_bytes (int, default 65536):
                Maximum file size in bytes for any single context block.
                Files exceeding this limit are replaced with a notice
                directing the model to use read_file.

            max_injection_tokens (int, default 32768):
                Maximum total estimated tokens for all dynamic context
                injections per turn (using len//4 approximation).
                Blocks beyond budget are skipped with a notice.

    Graceful degradation:
        If ``manifest_path`` is missing, the file does not exist, or the
        manifest contains no valid blocks, the hook logs a warning and
        returns without registering anything.  Everything else continues
        to work normally.
    """
    config = config or {}

    # ------------------------------------------------------------------ #
    # Resolve manifest path
    # ------------------------------------------------------------------ #
    manifest_path_raw: str | None = config.get("manifest_path")
    if not manifest_path_raw:
        logger.warning(
            "dynamic-context: 'manifest_path' not set in config — hook disabled. "
            "Add manifest_path to your module config."
        )
        return

    manifest_path = Path(manifest_path_raw)
    if not manifest_path.is_absolute():
        # Resolve relative paths against the session working directory
        working_dir_str: str = coordinator.get_capability("session.working_dir") or "."
        manifest_path = Path(working_dir_str) / manifest_path

    if not manifest_path.exists():
        logger.warning(
            "dynamic-context: Manifest not found at %s — hook disabled. "
            "Create the manifest file or fix the manifest_path config.",
            manifest_path,
        )
        return

    # ------------------------------------------------------------------ #
    # Load manifest
    # ------------------------------------------------------------------ #
    manifest = DynamicContextManifest(str(manifest_path))
    if not manifest.blocks:
        logger.warning(
            "dynamic-context: Manifest at %s loaded but contains no valid blocks — hook disabled.",
            manifest_path,
        )
        return

    # ------------------------------------------------------------------ #
    # Build shared state
    # ------------------------------------------------------------------ #
    classifier = ContextClassifier(manifest)
    content_cache: dict[str, str] = {}

    # Store on session_state so tool-load-context can access them
    if hasattr(coordinator, "session_state"):
        coordinator.session_state["dynamic_context_manifest"] = manifest
        coordinator.session_state["dynamic_context_classifier"] = classifier
        coordinator.session_state["dynamic_context_cache"] = content_cache

    # ------------------------------------------------------------------ #
    # Register hook
    # ------------------------------------------------------------------ #
    priority: int = int(config.get("priority", 20))
    max_file_bytes: int = int(config.get("max_file_bytes", 65536))
    max_injection_tokens: int = int(config.get("max_injection_tokens", 32768))
    loader = ContextLoaderHook(
        manifest,
        classifier,
        content_cache,
        priority=priority,
        max_file_bytes=max_file_bytes,
        max_injection_tokens=max_injection_tokens,
    )
    loader.register(coordinator.hooks)

    logger.info(
        "dynamic-context: Mounted — %d context blocks, priority=%d, max_file_bytes=%d, max_injection_tokens=%d, manifest=%s",
        len(manifest.blocks),
        priority,
        max_file_bytes,
        max_injection_tokens,
        manifest_path,
    )

    # Also mount the companion load_context tool so the LLM can
    # explicitly request context blocks.  Doing this here (instead of
    # via a separate module entry) avoids source-based validation issues
    # when both modules share the same Python package.
    try:
        from .tool import LoadContextTool

        tool = LoadContextTool(coordinator=coordinator)
        await coordinator.mount("tools", tool, name=tool.name)
        logger.info("dynamic-context: Mounted companion load_context tool")
    except Exception as tool_err:
        logger.warning(
            "dynamic-context: Failed to mount load_context tool: %s", tool_err
        )


__all__ = [
    "DynamicContextManifest",
    "LazyContextBlock",
    "ContextClassifier",
    "ContextLoaderHook",
    "mount",
]
