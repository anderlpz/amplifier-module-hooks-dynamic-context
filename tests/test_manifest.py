"""Tests for DynamicContextManifest and LazyContextBlock."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from amplifier_module_hooks_dynamic_context import DynamicContextManifest, LazyContextBlock


def write_manifest(tmp_path: Path, blocks: list[dict]) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.dump({"context_blocks": blocks}), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# LazyContextBlock
# ---------------------------------------------------------------------------


class TestLazyContextBlock:
    def test_load_reads_file_and_caches(self, tmp_path: Path) -> None:
        content_file = tmp_path / "kb.md"
        content_file.write_text("# Knowledge", encoding="utf-8")

        block = LazyContextBlock(name="kb", description="desc", path=str(content_file))
        cache: dict[str, str] = {}

        result = block.load(cache)

        assert result == "# Knowledge"
        assert block._content == "# Knowledge"
        assert str(content_file) in cache

    def test_load_uses_shared_cache(self, tmp_path: Path) -> None:
        content_file = tmp_path / "kb.md"
        content_file.write_text("original", encoding="utf-8")

        block = LazyContextBlock(name="kb", description="", path=str(content_file))
        cache = {str(content_file): "cached value"}

        result = block.load(cache)

        assert result == "cached value"  # Cache wins over disk

    def test_load_uses_instance_cache(self, tmp_path: Path) -> None:
        content_file = tmp_path / "kb.md"
        content_file.write_text("disk", encoding="utf-8")

        block = LazyContextBlock(name="kb", description="", path=str(content_file))
        block._content = "instance cache"
        cache: dict[str, str] = {}

        result = block.load(cache)

        assert result == "instance cache"

    def test_load_returns_none_on_missing_file(self, tmp_path: Path) -> None:
        block = LazyContextBlock(name="kb", description="", path="/nonexistent/path.md")
        cache: dict[str, str] = {}
        result = block.load(cache)
        assert result is None


# ---------------------------------------------------------------------------
# DynamicContextManifest
# ---------------------------------------------------------------------------


class TestDynamicContextManifest:
    def test_loads_valid_manifest(self, tmp_path: Path) -> None:
        content_file = tmp_path / "kb.md"
        content_file.write_text("content", encoding="utf-8")

        manifest_path = write_manifest(
            tmp_path,
            [
                {
                    "name": "imessage-knowledge",
                    "description": "iMessage queries",
                    "path": str(content_file),
                    "triggers": ["message", "text", "sms"],
                }
            ],
        )

        manifest = DynamicContextManifest(str(manifest_path))

        assert len(manifest.blocks) == 1
        block = manifest.blocks[0]
        assert block.name == "imessage-knowledge"
        assert block.description == "iMessage queries"
        assert block.triggers == ["message", "text", "sms"]

    def test_loads_multiple_blocks(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.md"
        f1.write_text("A", encoding="utf-8")
        f2 = tmp_path / "b.md"
        f2.write_text("B", encoding="utf-8")

        manifest_path = write_manifest(
            tmp_path,
            [
                {"name": "a", "description": "A block", "path": str(f1), "triggers": ["aa"]},
                {"name": "b", "description": "B block", "path": str(f2), "triggers": ["bb"]},
            ],
        )

        manifest = DynamicContextManifest(str(manifest_path))

        assert manifest.all_names() == ["a", "b"]

    def test_find_returns_correct_block(self, tmp_path: Path) -> None:
        f = tmp_path / "x.md"
        f.write_text("x", encoding="utf-8")

        manifest_path = write_manifest(
            tmp_path,
            [{"name": "x-block", "description": "X", "path": str(f), "triggers": ["x"]}],
        )

        manifest = DynamicContextManifest(str(manifest_path))

        assert manifest.find("x-block") is not None
        assert manifest.find("nonexistent") is None

    def test_graceful_on_missing_manifest(self, tmp_path: Path) -> None:
        manifest = DynamicContextManifest(str(tmp_path / "nonexistent.yaml"))
        assert manifest.blocks == []

    def test_graceful_on_invalid_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : not yaml", encoding="utf-8")
        manifest = DynamicContextManifest(str(bad))
        assert manifest.blocks == []

    def test_graceful_on_missing_context_blocks_key(self, tmp_path: Path) -> None:
        p = tmp_path / "m.yaml"
        p.write_text(yaml.dump({"other_key": []}), encoding="utf-8")
        manifest = DynamicContextManifest(str(p))
        assert manifest.blocks == []

    def test_skips_invalid_entries(self, tmp_path: Path) -> None:
        f = tmp_path / "valid.md"
        f.write_text("ok", encoding="utf-8")

        p = tmp_path / "m.yaml"
        p.write_text(
            yaml.dump(
                {
                    "context_blocks": [
                        {"name": "valid", "path": str(f), "triggers": ["v"]},
                        "not_a_dict",
                        {"description": "missing name and path"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        manifest = DynamicContextManifest(str(p))
        assert len(manifest.blocks) == 1
        assert manifest.blocks[0].name == "valid"
