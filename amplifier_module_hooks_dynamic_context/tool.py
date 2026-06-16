"""
load_context tool — explicit capability context loading.

Lets the LLM explicitly request a capability context block by name, or
list all blocks.  This is the manual fallback for cases where the
auto-classifier missed a relevant topic.

When load_context is called:
  1. The named block's content is returned directly in the tool result.
  2. The block's name is added to ``classifier.active_tags`` so it will
     be injected automatically on all subsequent turns too.

Requires hooks-dynamic-context to be mounted first — that hook populates
``session_state["dynamic_context_manifest"]`` and
``session_state["dynamic_context_classifier"]``.
"""

from __future__ import annotations

# Amplifier module metadata
__amplifier_module_type__ = "tool"

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import the real ToolResult; fall back to a minimal stub so the
# module can be imported in environments without amplifier_core installed.
try:
    from amplifier_core import ToolResult  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover

    class ToolResult:  # type: ignore[no-redef]
        """Minimal ToolResult stub for import-time safety."""

        def __init__(
            self,
            success: bool = True,
            output: Any = None,
            error: dict[str, Any] | None = None,
        ) -> None:
            self.success = success
            self.output = output
            self.error = error

        def __str__(self) -> str:
            if self.error:
                return str(self.error)
            return str(self.output) if self.output is not None else ""


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


class LoadContextTool:
    """
    Tool for explicitly loading a capability context block by name.

    Attributes:
        name:        Tool name exposed to the LLM.
        description: Natural-language description sent in the tool schema.
    """

    name = "load_context"
    description = (
        "Load a capability context block by name. Use when you need knowledge that "
        "wasn't automatically loaded. Call with the name from the available capability "
        "context list. Use name='list' to show all available blocks with descriptions."
    )

    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Name of the context block to load, or 'list' to show "
                        "all available blocks with descriptions and trigger keywords."
                    ),
                },
            },
            "required": ["name"],
        }

    # ------------------------------------------------------------------
    # Accessors for shared state written by the hook
    # ------------------------------------------------------------------

    def _get_manifest(self) -> Any:
        """Return DynamicContextManifest from session_state, or None."""
        return self._session_state().get("dynamic_context_manifest")

    def _get_classifier(self) -> Any:
        """Return ContextClassifier from session_state, or None."""
        return self._session_state().get("dynamic_context_classifier")

    def _get_cache(self) -> dict[str, str]:
        """Return the shared content cache from session_state."""
        return self._session_state().get("dynamic_context_cache", {})

    def _session_state(self) -> dict[str, Any]:
        return getattr(self.coordinator, "session_state", {}) or {}

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Dispatch to list or load handler."""
        name = str(input.get("name", "")).strip()

        manifest = self._get_manifest()
        if manifest is None:
            return ToolResult(
                success=False,
                error={
                    "code": "not_mounted",
                    "message": (
                        "hooks-dynamic-context is not mounted. "
                        "Add it to your bundle configuration with a manifest_path "
                        "setting pointing to your capability manifest YAML."
                    ),
                },
            )

        if name == "list":
            return self._handle_list(manifest)

        if not name:
            return ToolResult(
                success=False,
                error={
                    "code": "missing_name",
                    "message": "The 'name' parameter is required. Use name='list' to see available blocks.",
                },
            )

        return self._handle_load(name, manifest)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_list(self, manifest: Any) -> ToolResult:
        """List all available context blocks with status."""
        classifier = self._get_classifier()
        active_tags: set[str] = (
            classifier.active_tags if classifier is not None else set()
        )

        blocks = []
        for block in manifest.blocks:
            blocks.append(
                {
                    "name": block.name,
                    "description": block.description,
                    "triggers": block.triggers,
                    "loaded": block.name in active_tags,
                    "path": block.path,
                }
            )

        return ToolResult(
            success=True,
            output={
                "available_blocks": blocks,
                "loaded_count": len(active_tags),
                "total_count": len(blocks),
                "tip": "Call load_context with a block name to load it explicitly.",
            },
        )

    def _handle_load(self, name: str, manifest: Any) -> ToolResult:
        """Load a specific context block, inject into classifier, return content."""
        block = manifest.find(name)
        if block is None:
            available = manifest.all_names()
            return ToolResult(
                success=False,
                error={
                    "code": "block_not_found",
                    "message": (
                        f"Context block '{name}' not found in manifest. "
                        f"Use name='list' to see all available blocks."
                    ),
                    "available": available,
                },
            )

        # Use the shared cache so we don't re-read files already loaded by the hook
        cache = self._get_cache()
        content = block.load(cache)
        if content is None:
            return ToolResult(
                success=False,
                error={
                    "code": "load_failed",
                    "message": (
                        f"Failed to read content for block '{name}' from {block.path}. "
                        f"Check that the file exists and is readable."
                    ),
                },
            )

        # Activate the tag so the hook injects this block on all future turns
        classifier = self._get_classifier()
        if classifier is not None:
            classifier.activate(name)

        logger.info(
            "load_context: Loaded '%s' (%d chars), added to active_tags",
            name,
            len(content),
        )

        return ToolResult(
            success=True,
            output={
                "name": block.name,
                "description": block.description,
                "content": content,
                "characters": len(content),
                "message": (
                    f"Loaded '{name}'. "
                    f"This context will now be injected automatically on all subsequent turns."
                ),
            },
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Mount the load_context tool.

    Should be declared alongside (or after) hooks-dynamic-context in your
    bundle so the shared session_state is already populated by the time
    the tool is first called.  The tool validates at first use if the hook
    is not yet mounted.

    Args:
        coordinator: Module coordinator provided by the Amplifier runtime.
        config:      Unused; reserved for future configuration.
    """
    tool = LoadContextTool(coordinator=coordinator)
    await coordinator.mount("tools", tool, name=tool.name)
    logger.info("dynamic-context: Mounted load_context tool")
    return {
        "name": "tool-load-context",
        "version": "0.1.0",
        "provides": ["load_context"],
    }
