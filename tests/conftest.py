"""Shared test fixtures for hooks-dynamic-context tests."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_manifest(tmp_path: Path, blocks: list[dict]) -> Path:
    """Write a temporary manifest YAML and return its path."""
    data = {"context_blocks": blocks}
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.dump(data), encoding="utf-8")
    return manifest_path


def write_content_file(tmp_path: Path, name: str, content: str) -> Path:
    """Write a temporary context content file and return its path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Mock coordinator
# ---------------------------------------------------------------------------


@pytest.fixture
def coordinator():
    """Minimal mock coordinator with session_state and hooks."""
    mock = MagicMock()
    mock.session_state = {}
    mock.hooks = MagicMock()
    mock.get_capability = MagicMock(return_value=None)
    return mock
