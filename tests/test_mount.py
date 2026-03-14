"""Tests for the mount() entry point — graceful degradation and wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import yaml

from amplifier_module_hooks_dynamic_context import mount


def make_coordinator() -> MagicMock:
    mock = MagicMock()
    mock.session_state = {}
    mock.hooks = MagicMock()
    mock.get_capability = MagicMock(return_value=None)
    return mock


def write_manifest(tmp_path: Path, blocks: list[dict]) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.dump({"context_blocks": blocks}), encoding="utf-8")
    return p


class TestMountGracefulDegradation:
    @pytest.mark.asyncio
    async def test_no_manifest_path_does_not_register(self) -> None:
        coordinator = make_coordinator()
        await mount(coordinator, config={})
        coordinator.hooks.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonexistent_manifest_does_not_register(self, tmp_path: Path) -> None:
        coordinator = make_coordinator()
        await mount(coordinator, config={"manifest_path": str(tmp_path / "missing.yaml")})
        coordinator.hooks.register.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_manifest_does_not_register(self, tmp_path: Path) -> None:
        p = write_manifest(tmp_path, [])  # No blocks
        coordinator = make_coordinator()
        await mount(coordinator, config={"manifest_path": str(p)})
        coordinator.hooks.register.assert_not_called()


class TestMountWiring:
    @pytest.mark.asyncio
    async def test_registers_hook_on_provider_request(self, tmp_path: Path) -> None:
        content = tmp_path / "kb.md"
        content.write_text("knowledge", encoding="utf-8")
        p = write_manifest(
            tmp_path,
            [{"name": "kb", "description": "Knowledge base", "path": str(content), "triggers": ["k"]}],
        )

        coordinator = make_coordinator()
        await mount(coordinator, config={"manifest_path": str(p)})

        coordinator.hooks.register.assert_called_once()
        args, kwargs = coordinator.hooks.register.call_args
        assert args[0] == "provider:request"
        assert kwargs.get("name") == "hooks-dynamic-context"

    @pytest.mark.asyncio
    async def test_stores_shared_state_in_session_state(self, tmp_path: Path) -> None:
        content = tmp_path / "kb.md"
        content.write_text("knowledge", encoding="utf-8")
        p = write_manifest(
            tmp_path,
            [{"name": "kb", "description": "KB", "path": str(content), "triggers": ["k"]}],
        )

        coordinator = make_coordinator()
        await mount(coordinator, config={"manifest_path": str(p)})

        assert "dynamic_context_manifest" in coordinator.session_state
        assert "dynamic_context_classifier" in coordinator.session_state
        assert "dynamic_context_cache" in coordinator.session_state

    @pytest.mark.asyncio
    async def test_custom_priority_passed_through(self, tmp_path: Path) -> None:
        content = tmp_path / "kb.md"
        content.write_text("knowledge", encoding="utf-8")
        p = write_manifest(
            tmp_path,
            [{"name": "kb", "description": "KB", "path": str(content), "triggers": ["k"]}],
        )

        coordinator = make_coordinator()
        await mount(coordinator, config={"manifest_path": str(p), "priority": 50})

        _, kwargs = coordinator.hooks.register.call_args
        assert kwargs.get("priority") == 50

    @pytest.mark.asyncio
    async def test_default_priority_is_20(self, tmp_path: Path) -> None:
        content = tmp_path / "kb.md"
        content.write_text("knowledge", encoding="utf-8")
        p = write_manifest(
            tmp_path,
            [{"name": "kb", "description": "KB", "path": str(content), "triggers": ["k"]}],
        )

        coordinator = make_coordinator()
        await mount(coordinator, config={"manifest_path": str(p)})

        _, kwargs = coordinator.hooks.register.call_args
        assert kwargs.get("priority") == 20

    @pytest.mark.asyncio
    async def test_relative_manifest_path_resolved_via_capability(self, tmp_path: Path) -> None:
        content = tmp_path / "kb.md"
        content.write_text("knowledge", encoding="utf-8")
        p = write_manifest(
            tmp_path,
            [{"name": "kb", "description": "KB", "path": str(content), "triggers": ["k"]}],
        )

        coordinator = make_coordinator()
        # Return tmp_path as the working_dir capability
        coordinator.get_capability = MagicMock(return_value=str(tmp_path))

        # Use relative path (just the filename)
        await mount(coordinator, config={"manifest_path": p.name})

        # Should have successfully loaded and registered
        coordinator.hooks.register.assert_called_once()

    @pytest.mark.asyncio
    async def test_null_config_treated_as_empty(self) -> None:
        coordinator = make_coordinator()
        # Should not raise
        await mount(coordinator, config=None)
        coordinator.hooks.register.assert_not_called()
