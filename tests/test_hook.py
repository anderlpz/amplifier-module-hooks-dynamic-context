"""Tests for ContextLoaderHook and the provider:request injection path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from amplifier_module_hooks_dynamic_context import (
    ContextClassifier,
    ContextLoaderHook,
    DynamicContextManifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_manifest_and_files(tmp_path: Path) -> tuple[DynamicContextManifest, dict[str, Path]]:
    files = {
        "imessage": tmp_path / "imessage.md",
        "calendar": tmp_path / "calendar.md",
    }
    files["imessage"].write_text("# iMessage Knowledge\nHow to query messages.", encoding="utf-8")
    files["calendar"].write_text("# Calendar Knowledge\nHow to manage events.", encoding="utf-8")

    p = tmp_path / "manifest.yaml"
    p.write_text(
        yaml.dump(
            {
                "context_blocks": [
                    {
                        "name": "imessage-knowledge",
                        "description": "iMessage queries and AppleScript",
                        "path": str(files["imessage"]),
                        "triggers": ["message", "text", "imessage", "sms"],
                    },
                    {
                        "name": "calendar-knowledge",
                        "description": "Calendar API and event management",
                        "path": str(files["calendar"]),
                        "triggers": ["calendar", "schedule", "meeting"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return DynamicContextManifest(str(p)), files


@pytest.fixture
def manifest_and_files(tmp_path: Path):
    return make_manifest_and_files(tmp_path)


# ---------------------------------------------------------------------------
# Thin manifest injection
# ---------------------------------------------------------------------------


class TestThinManifestAlwaysInjected:
    @pytest.mark.asyncio
    async def test_thin_manifest_present_with_no_active_tags(
        self, manifest_and_files: tuple
    ) -> None:
        manifest, _ = manifest_and_files
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        # Message that doesn't trigger anything
        data = {"messages": [{"role": "user", "content": "hello world"}]}
        result = await hook.on_provider_request("provider:request", data)

        assert result.action == "inject_context"
        assert 'source="dynamic-context-manifest"' in result.context_injection
        assert "imessage-knowledge" in result.context_injection
        assert "calendar-knowledge" in result.context_injection
        # No content blocks loaded yet
        assert 'source="dynamic-context:' not in result.context_injection

    @pytest.mark.asyncio
    async def test_loaded_marker_shown_for_active_blocks(self, manifest_and_files: tuple) -> None:
        manifest, _ = manifest_and_files
        classifier = ContextClassifier(manifest)
        classifier.activate("imessage-knowledge")
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        data = {"messages": [{"role": "user", "content": "neutral message"}]}
        result = await hook.on_provider_request("provider:request", data)

        assert "imessage-knowledge: iMessage queries and AppleScript [loaded]" in result.context_injection
        assert "[loaded]" not in result.context_injection.replace(
            "imessage-knowledge: iMessage queries and AppleScript [loaded]", ""
        ).replace("calendar-knowledge:", "")[: result.context_injection.find("calendar-knowledge:")]


# ---------------------------------------------------------------------------
# Content injection
# ---------------------------------------------------------------------------


class TestContentInjection:
    @pytest.mark.asyncio
    async def test_content_injected_when_block_activated(self, manifest_and_files: tuple) -> None:
        manifest, _ = manifest_and_files
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        # Trigger iMessage block
        data = {"messages": [{"role": "user", "content": "send a text message to Bob"}]}
        result = await hook.on_provider_request("provider:request", data)

        assert 'source="dynamic-context:imessage-knowledge"' in result.context_injection
        assert "# iMessage Knowledge" in result.context_injection
        assert 'source="dynamic-context:calendar-knowledge"' not in result.context_injection

    @pytest.mark.asyncio
    async def test_multiple_blocks_injected(self, manifest_and_files: tuple) -> None:
        manifest, _ = manifest_and_files
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        data = {
            "messages": [
                {"role": "user", "content": "schedule a meeting and send a text to confirm"}
            ]
        }
        result = await hook.on_provider_request("provider:request", data)

        assert 'source="dynamic-context:imessage-knowledge"' in result.context_injection
        assert 'source="dynamic-context:calendar-knowledge"' in result.context_injection

    @pytest.mark.asyncio
    async def test_content_cached_not_reread(self, tmp_path: Path) -> None:
        """After first load, the file should not be read from disk again."""
        manifest, files = make_manifest_and_files(tmp_path)
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        # Activate the block
        data = {"messages": [{"role": "user", "content": "send a message"}]}
        await hook.on_provider_request("provider:request", data)

        # Overwrite the file on disk — should NOT affect output (cached)
        files["imessage"].write_text("OVERWRITTEN", encoding="utf-8")

        result = await hook.on_provider_request("provider:request", {"messages": []})
        assert "OVERWRITTEN" not in result.context_injection
        assert "# iMessage Knowledge" in result.context_injection


# ---------------------------------------------------------------------------
# HookResult properties
# ---------------------------------------------------------------------------


class TestHookResultProperties:
    @pytest.mark.asyncio
    async def test_result_is_ephemeral_and_suppressed(self, manifest_and_files: tuple) -> None:
        manifest, _ = manifest_and_files
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        data = {"messages": [{"role": "user", "content": "hello"}]}
        result = await hook.on_provider_request("provider:request", data)

        assert result.ephemeral is True
        assert result.suppress_output is True
        assert result.context_injection_role == "user"

    @pytest.mark.asyncio
    async def test_returns_continue_when_no_blocks(self, tmp_path: Path) -> None:
        """If manifest has no blocks, hook should return continue, not inject."""
        p = tmp_path / "empty.yaml"
        p.write_text(yaml.dump({"context_blocks": []}), encoding="utf-8")
        manifest = DynamicContextManifest(str(p))
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        data = {"messages": [{"role": "user", "content": "some text"}]}
        result = await hook.on_provider_request("provider:request", data)

        assert result.action == "continue"


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


class TestMessageExtraction:
    @pytest.mark.asyncio
    async def test_extracts_last_user_message(self, manifest_and_files: tuple) -> None:
        manifest, _ = manifest_and_files
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        data = {
            "messages": [
                {"role": "user", "content": "first message — no triggers"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "send a text to Carol"},  # Last user message
            ]
        }
        result = await hook.on_provider_request("provider:request", data)

        assert 'source="dynamic-context:imessage-knowledge"' in result.context_injection

    @pytest.mark.asyncio
    async def test_handles_structured_content_blocks(self, manifest_and_files: tuple) -> None:
        manifest, _ = manifest_and_files
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "send a text message to Alice"}
                    ],
                }
            ]
        }
        result = await hook.on_provider_request("provider:request", data)

        assert 'source="dynamic-context:imessage-knowledge"' in result.context_injection

    @pytest.mark.asyncio
    async def test_handles_empty_messages_list(self, manifest_and_files: tuple) -> None:
        manifest, _ = manifest_and_files
        classifier = ContextClassifier(manifest)
        cache: dict[str, str] = {}
        hook = ContextLoaderHook(manifest, classifier, cache)

        # No exception with empty messages
        result = await hook.on_provider_request("provider:request", {"messages": []})
        assert result.action == "inject_context"
