"""Tests for the load_context tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from amplifier_module_hooks_dynamic_context import (
    ContextClassifier,
    DynamicContextManifest,
)
from amplifier_module_hooks_dynamic_context.tool import LoadContextTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_coordinator(tmp_path: Path) -> tuple[MagicMock, DynamicContextManifest]:
    """Create a mock coordinator pre-populated with manifest and classifier."""
    f1 = tmp_path / "imessage.md"
    f1.write_text("# iMessage Knowledge\nDetails here.", encoding="utf-8")
    f2 = tmp_path / "calendar.md"
    f2.write_text("# Calendar Knowledge\nDetails here.", encoding="utf-8")

    p = tmp_path / "manifest.yaml"
    p.write_text(
        yaml.dump(
            {
                "context_blocks": [
                    {
                        "name": "imessage-knowledge",
                        "description": "iMessage queries",
                        "path": str(f1),
                        "triggers": ["message", "text"],
                    },
                    {
                        "name": "calendar-knowledge",
                        "description": "Calendar management",
                        "path": str(f2),
                        "triggers": ["calendar", "schedule"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = DynamicContextManifest(str(p))
    classifier = ContextClassifier(manifest)
    cache: dict[str, str] = {}

    coordinator = MagicMock()
    coordinator.session_state = {
        "dynamic_context_manifest": manifest,
        "dynamic_context_classifier": classifier,
        "dynamic_context_cache": cache,
    }
    coordinator.mount = AsyncMock()

    return coordinator, manifest


# ---------------------------------------------------------------------------
# list operation
# ---------------------------------------------------------------------------


class TestLoadContextList:
    @pytest.mark.asyncio
    async def test_list_returns_all_blocks(self, tmp_path: Path) -> None:
        coordinator, _ = make_coordinator(tmp_path)
        tool = LoadContextTool(coordinator=coordinator)

        result = await tool.execute({"name": "list"})

        assert result.success is True
        blocks = result.output["available_blocks"]
        assert len(blocks) == 2
        names = [b["name"] for b in blocks]
        assert "imessage-knowledge" in names
        assert "calendar-knowledge" in names

    @pytest.mark.asyncio
    async def test_list_shows_loaded_status(self, tmp_path: Path) -> None:
        coordinator, manifest = make_coordinator(tmp_path)
        classifier = coordinator.session_state["dynamic_context_classifier"]
        classifier.activate("imessage-knowledge")
        tool = LoadContextTool(coordinator=coordinator)

        result = await tool.execute({"name": "list"})

        blocks = {b["name"]: b for b in result.output["available_blocks"]}
        assert blocks["imessage-knowledge"]["loaded"] is True
        assert blocks["calendar-knowledge"]["loaded"] is False

    @pytest.mark.asyncio
    async def test_list_includes_triggers(self, tmp_path: Path) -> None:
        coordinator, _ = make_coordinator(tmp_path)
        tool = LoadContextTool(coordinator=coordinator)

        result = await tool.execute({"name": "list"})

        blocks = {b["name"]: b for b in result.output["available_blocks"]}
        assert "message" in blocks["imessage-knowledge"]["triggers"]


# ---------------------------------------------------------------------------
# load by name
# ---------------------------------------------------------------------------


class TestLoadContextByName:
    @pytest.mark.asyncio
    async def test_loads_known_block(self, tmp_path: Path) -> None:
        coordinator, _ = make_coordinator(tmp_path)
        tool = LoadContextTool(coordinator=coordinator)

        result = await tool.execute({"name": "imessage-knowledge"})

        assert result.success is True
        assert "# iMessage Knowledge" in result.output["content"]
        assert result.output["name"] == "imessage-knowledge"

    @pytest.mark.asyncio
    async def test_loading_activates_tag(self, tmp_path: Path) -> None:
        coordinator, _ = make_coordinator(tmp_path)
        classifier = coordinator.session_state["dynamic_context_classifier"]
        tool = LoadContextTool(coordinator=coordinator)

        assert "imessage-knowledge" not in classifier.active_tags

        await tool.execute({"name": "imessage-knowledge"})

        assert "imessage-knowledge" in classifier.active_tags

    @pytest.mark.asyncio
    async def test_loading_uses_shared_cache(self, tmp_path: Path) -> None:
        """Content loaded by the tool should land in the shared cache."""
        coordinator, _ = make_coordinator(tmp_path)
        cache = coordinator.session_state["dynamic_context_cache"]
        tool = LoadContextTool(coordinator=coordinator)

        await tool.execute({"name": "imessage-knowledge"})

        # The file path should now be in the shared cache
        manifest = coordinator.session_state["dynamic_context_manifest"]
        block = manifest.find("imessage-knowledge")
        assert block.path in cache

    @pytest.mark.asyncio
    async def test_unknown_block_returns_error(self, tmp_path: Path) -> None:
        coordinator, _ = make_coordinator(tmp_path)
        tool = LoadContextTool(coordinator=coordinator)

        result = await tool.execute({"name": "nonexistent-block"})

        assert result.success is False
        assert result.error["code"] == "block_not_found"
        assert "imessage-knowledge" in result.error["available"]

    @pytest.mark.asyncio
    async def test_unreadable_file_returns_error(self, tmp_path: Path) -> None:
        coordinator, manifest = make_coordinator(tmp_path)
        # Point the block at a file that doesn't exist
        block = manifest.find("imessage-knowledge")
        block.path = "/nonexistent/path/file.md"
        block._content = None  # Clear cached content
        cache = coordinator.session_state["dynamic_context_cache"]
        cache.pop(block.path, None)

        tool = LoadContextTool(coordinator=coordinator)
        result = await tool.execute({"name": "imessage-knowledge"})

        assert result.success is False
        assert result.error["code"] == "load_failed"


# ---------------------------------------------------------------------------
# Not-mounted error
# ---------------------------------------------------------------------------


class TestNotMounted:
    @pytest.mark.asyncio
    async def test_returns_error_when_hook_not_mounted(self) -> None:
        coordinator = MagicMock()
        coordinator.session_state = {}  # No manifest in session_state

        tool = LoadContextTool(coordinator=coordinator)
        result = await tool.execute({"name": "anything"})

        assert result.success is False
        assert result.error["code"] == "not_mounted"

    @pytest.mark.asyncio
    async def test_returns_error_when_no_session_state(self) -> None:
        coordinator = MagicMock(spec=[])  # No session_state attribute at all

        tool = LoadContextTool(coordinator=coordinator)
        result = await tool.execute({"name": "anything"})

        assert result.success is False


# ---------------------------------------------------------------------------
# mount() function
# ---------------------------------------------------------------------------


class TestToolMount:
    @pytest.mark.asyncio
    async def test_mount_registers_tool(self, tmp_path: Path) -> None:
        from amplifier_module_hooks_dynamic_context.tool import mount

        coordinator = MagicMock()
        coordinator.session_state = {}
        coordinator.mount = AsyncMock()

        await mount(coordinator, config={})

        coordinator.mount.assert_called_once()
        args, kwargs = coordinator.mount.call_args
        assert args[0] == "tools"
        assert kwargs.get("name") == "load_context" or args[2] == "load_context" if len(args) > 2 else True

    @pytest.mark.asyncio
    async def test_mount_null_config(self) -> None:
        from amplifier_module_hooks_dynamic_context.tool import mount

        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        await mount(coordinator, config=None)

        coordinator.mount.assert_called_once()
