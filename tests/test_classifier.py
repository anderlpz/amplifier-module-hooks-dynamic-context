"""Tests for ContextClassifier."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from amplifier_module_hooks_dynamic_context import ContextClassifier, DynamicContextManifest


def make_manifest(tmp_path: Path) -> DynamicContextManifest:
    f1 = tmp_path / "imessage.md"
    f1.write_text("iMessage content", encoding="utf-8")
    f2 = tmp_path / "calendar.md"
    f2.write_text("Calendar content", encoding="utf-8")
    f3 = tmp_path / "email.md"
    f3.write_text("Email content", encoding="utf-8")

    p = tmp_path / "manifest.yaml"
    p.write_text(
        yaml.dump(
            {
                "context_blocks": [
                    {
                        "name": "imessage-knowledge",
                        "description": "iMessage",
                        "path": str(f1),
                        "triggers": ["message", "text", "imessage", "sms", "texted"],
                    },
                    {
                        "name": "calendar-knowledge",
                        "description": "Calendar",
                        "path": str(f2),
                        "triggers": ["calendar", "schedule", "meeting"],
                    },
                    {
                        "name": "email-knowledge",
                        "description": "Email",
                        "path": str(f3),
                        "triggers": ["send an email", "email thread"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return DynamicContextManifest(str(p))


class TestContextClassifier:
    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)
        result = classifier.classify("tell me about the weather")
        assert result == set()

    def test_single_word_trigger_whole_word_match(self, tmp_path: Path) -> None:
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        result = classifier.classify("send a message to Alice")
        assert "imessage-knowledge" in result

    def test_single_word_trigger_no_partial_match(self, tmp_path: Path) -> None:
        """'text' should not match inside 'context' as a substring."""
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        # 'text' is a trigger but 'context' contains 'text' as substring
        # Word-boundary matching should NOT fire here
        result = classifier.classify("here is the full context of my question")
        # 'text' IS in 'context' as substring, but NOT as whole word
        assert "imessage-knowledge" not in result

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        result = classifier.classify("Can you check my IMESSAGE inbox?")
        assert "imessage-knowledge" in result

    def test_compound_trigger_substring_match(self, tmp_path: Path) -> None:
        """Multi-word trigger uses substring match."""
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        result = classifier.classify("I need to send an email to my team")
        assert "email-knowledge" in result

    def test_multiple_blocks_match(self, tmp_path: Path) -> None:
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        result = classifier.classify("schedule a meeting and text Bob")
        assert "calendar-knowledge" in result
        assert "imessage-knowledge" in result

    def test_tags_persist_across_calls(self, tmp_path: Path) -> None:
        """Once activated, a tag stays active even if not triggered again."""
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        classifier.classify("send a text message")
        result = classifier.classify("what is 2 + 2?")  # No triggers

        assert "imessage-knowledge" in result

    def test_no_double_activation(self, tmp_path: Path) -> None:
        """Classifying the same trigger twice doesn't break anything."""
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        classifier.classify("send a text")
        classifier.classify("send another text")

        assert len(classifier.active_tags) == 1  # Not duplicated

    def test_manual_activate(self, tmp_path: Path) -> None:
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        classifier.activate("calendar-knowledge")
        assert "calendar-knowledge" in classifier.active_tags

    def test_empty_message(self, tmp_path: Path) -> None:
        manifest = make_manifest(tmp_path)
        classifier = ContextClassifier(manifest)

        result = classifier.classify("")
        assert result == set()
