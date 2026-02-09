"""
Tests for PromptComposer and Canonicalizer.

These tests verify the CORRECTNESS requirement:
Equivalent prompts MUST produce identical tokens.

Run with: pytest tests/test_composer.py -v
"""

import pytest

from src.canonicalizer import Canonicalizer, CanonicalizeConfig, canonicalize_text
from src.composer import PromptComposer, create_simple_prompt
from src.config import Segment, SegmentType, ComposedPrompt


class TestCanonicalizer:
    """Tests for text canonicalization."""

    def test_whitespace_normalization(self):
        """Multiple spaces should collapse to single space."""
        canon = Canonicalizer()

        assert canon.canonicalize("hello  world") == "hello world"
        assert canon.canonicalize("hello   world") == "hello world"
        assert canon.canonicalize("hello\t\tworld") == "hello world"

    def test_strip(self):
        """Leading/trailing whitespace should be removed."""
        canon = Canonicalizer()

        assert canon.canonicalize("  hello  ") == "hello"
        assert canon.canonicalize("\n\nhello\n\n") == "hello"

    def test_newline_normalization(self):
        """Excessive newlines should collapse to max 2."""
        canon = Canonicalizer()

        result = canon.canonicalize("hello\n\n\n\nworld")
        assert result == "hello\n\nworld"

    def test_preserves_single_newlines(self):
        """Single newlines should be preserved (semantic meaning)."""
        canon = Canonicalizer()

        result = canon.canonicalize("line1\nline2")
        assert result == "line1\nline2"

    def test_json_normalization_object(self):
        """JSON objects should be normalized (sorted keys, no spaces)."""
        canon = Canonicalizer()

        # Different formatting, same content
        input1 = '{"b": 2, "a": 1}'
        input2 = '{"a":1,"b":2}'

        result1 = canon.canonicalize(input1)
        result2 = canon.canonicalize(input2)

        assert result1 == result2
        assert result1 == '{"a":1,"b":2}'

    def test_json_normalization_nested(self):
        """Nested JSON should be normalized."""
        canon = Canonicalizer()

        input_text = '{"outer": {"b": 2, "a": 1}}'
        result = canon.canonicalize(input_text)

        assert result == '{"outer":{"a":1,"b":2}}'

    def test_json_in_text(self):
        """JSON embedded in text should be normalized."""
        canon = Canonicalizer()

        input_text = 'Config: {"b": 2, "a": 1} end'
        result = canon.canonicalize(input_text)

        assert result == 'Config: {"a":1,"b":2} end'

    def test_invalid_json_unchanged(self):
        """Invalid JSON should be left unchanged."""
        canon = Canonicalizer()

        input_text = '{not valid json}'
        result = canon.canonicalize(input_text)

        assert '{not valid json}' in result

    def test_unicode_normalization(self):
        """Unicode should be normalized to NFC form."""
        canon = Canonicalizer()

        # e + combining accent vs precomposed é
        decomposed = "e\u0301"  # e + combining acute accent
        composed = "\u00e9"     # precomposed é

        result1 = canon.canonicalize(decomposed)
        result2 = canon.canonicalize(composed)

        assert result1 == result2
        assert result1 == composed

    def test_empty_string(self):
        """Empty string should return empty string."""
        canon = Canonicalizer()
        assert canon.canonicalize("") == ""

    def test_config_disable_whitespace(self):
        """Can disable whitespace normalization."""
        config = CanonicalizeConfig(normalize_whitespace=False, strip_segments=False)
        canon = Canonicalizer(config)

        result = canon.canonicalize("hello  world")
        assert result == "hello  world"

    def test_config_disable_json(self):
        """Can disable JSON normalization."""
        config = CanonicalizeConfig(normalize_json=False)
        canon = Canonicalizer(config)

        input_text = '{"b": 2, "a": 1}'
        result = canon.canonicalize(input_text)

        # Should still strip but not reorder
        assert '"b": 2' in result or '"b":2' in result

    def test_determinism(self):
        """Same input should always produce same output."""
        canon = Canonicalizer()

        input_text = '  Hello   {"b":2,"a":1}  world  '

        results = [canon.canonicalize(input_text) for _ in range(100)]

        assert len(set(results)) == 1  # All results identical


class TestPromptComposer:
    """Tests for prompt composition."""

    def test_simple_composition(self):
        """Basic segment composition."""
        composer = PromptComposer()

        segments = [
            Segment(SegmentType.SYSTEM, "You are helpful."),
            Segment(SegmentType.USER, "Hello!"),
        ]

        result = composer.compose(segments)

        assert "You are helpful." in result.text
        assert "Hello!" in result.text
        assert len(result.segment_boundaries) == 2

    def test_segment_ordering_validation(self):
        """Invalid segment order should raise error."""
        composer = PromptComposer()

        # USER before SYSTEM should fail
        segments = [
            Segment(SegmentType.USER, "Hello!"),
            Segment(SegmentType.SYSTEM, "You are helpful."),
        ]

        with pytest.raises(ValueError, match="Invalid segment order"):
            composer.compose(segments)

    def test_valid_ordering(self):
        """Valid ordering should work."""
        composer = PromptComposer()

        # SYSTEM -> TEMPLATE -> USER is valid
        segments = [
            Segment(SegmentType.SYSTEM, "System"),
            Segment(SegmentType.TEMPLATE, "Template"),
            Segment(SegmentType.USER, "User"),
        ]

        result = composer.compose(segments)
        assert len(result.segment_boundaries) == 3

    def test_same_type_multiple_times(self):
        """Multiple segments of same type should work."""
        composer = PromptComposer()

        segments = [
            Segment(SegmentType.SYSTEM, "System 1"),
            Segment(SegmentType.SYSTEM, "System 2"),
            Segment(SegmentType.USER, "User"),
        ]

        result = composer.compose(segments)
        assert "System 1" in result.text
        assert "System 2" in result.text

    def test_cacheable_prefix(self):
        """Cacheable prefix should include SYSTEM and TEMPLATE only."""
        composer = PromptComposer()

        segments = [
            Segment(SegmentType.SYSTEM, "System"),
            Segment(SegmentType.TEMPLATE, "Template"),
            Segment(SegmentType.USER, "User"),
        ]

        result = composer.compose(segments)

        assert "System" in result.cacheable_prefix
        assert "Template" in result.cacheable_prefix
        assert "User" not in result.cacheable_prefix

    def test_canonicalization_applied(self):
        """Composer should canonicalize segment content."""
        composer = PromptComposer()

        segments = [
            Segment(SegmentType.SYSTEM, "  Hello   world  "),
        ]

        result = composer.compose(segments)

        # Should be canonicalized (stripped, spaces collapsed)
        assert result.text == "Hello world"

    def test_segment_boundaries_correct(self):
        """Segment boundaries should have correct offsets."""
        composer = PromptComposer()

        segments = [
            Segment(SegmentType.SYSTEM, "ABC"),
            Segment(SegmentType.USER, "XYZ"),
        ]

        result = composer.compose(segments)

        # First segment: starts at 0
        start1, end1, type1 = result.segment_boundaries[0]
        assert start1 == 0
        assert end1 == 3  # "ABC"
        assert type1 == SegmentType.SYSTEM

        # Second segment: after separator
        start2, end2, type2 = result.segment_boundaries[1]
        assert type2 == SegmentType.USER
        assert result.text[start2:end2] == "XYZ"

    def test_empty_segments(self):
        """Empty segment list should return empty prompt."""
        composer = PromptComposer()

        result = composer.compose([])

        assert result.text == ""
        assert result.segment_boundaries == []

    def test_prefix_hash_determinism(self):
        """Same input should always produce same hash."""
        composer = PromptComposer()

        segments = [
            Segment(SegmentType.SYSTEM, "System prompt"),
            Segment(SegmentType.USER, "User query"),
        ]

        result = composer.compose(segments)

        hashes = [
            composer.compute_prefix_hash(result, "tenant1", "model1")
            for _ in range(100)
        ]

        assert len(set(hashes)) == 1

    def test_prefix_hash_tenant_isolation(self):
        """Different tenants should get different hashes."""
        composer = PromptComposer()

        segments = [
            Segment(SegmentType.SYSTEM, "Same system prompt"),
            Segment(SegmentType.USER, "Same user query"),
        ]

        result = composer.compose(segments)

        hash1 = composer.compute_prefix_hash(result, "tenant1", "model1")
        hash2 = composer.compute_prefix_hash(result, "tenant2", "model1")

        assert hash1 != hash2

    def test_prefix_hash_model_isolation(self):
        """Different models should get different hashes."""
        composer = PromptComposer()

        segments = [
            Segment(SegmentType.SYSTEM, "Same system prompt"),
        ]

        result = composer.compose(segments)

        hash1 = composer.compute_prefix_hash(result, "tenant1", "llama-7b")
        hash2 = composer.compute_prefix_hash(result, "tenant1", "llama-13b")

        assert hash1 != hash2

    def test_segment_metadata_affects_hash(self):
        """Segment metadata should affect the hash."""
        composer = PromptComposer()

        seg1 = Segment(SegmentType.SYSTEM, "System", metadata={"version": "1"})
        seg2 = Segment(SegmentType.SYSTEM, "System", metadata={"version": "2"})

        result1 = composer.compose([seg1])
        result2 = composer.compose([seg2])

        hash1 = composer.compute_prefix_hash(result1)
        hash2 = composer.compute_prefix_hash(result2)

        assert hash1 != hash2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_canonicalize_text(self):
        """Standalone canonicalize_text function."""
        result = canonicalize_text("  hello   world  ")
        assert result == "hello world"

    def test_create_simple_prompt(self):
        """Convenience function for simple prompts."""
        result = create_simple_prompt(
            system="You are helpful.",
            user="Hi!",
        )

        assert "You are helpful." in result.text
        assert "Hi!" in result.text
        assert isinstance(result, ComposedPrompt)

    def test_create_simple_prompt_with_template(self):
        """Simple prompt with template."""
        result = create_simple_prompt(
            system="System",
            user="User",
            template="Template",
        )

        assert "Template" in result.text
        assert "Template" in result.cacheable_prefix


class TestEquivalentPromptsDeterminism:
    """
    CRITICAL TEST: Equivalent prompts must produce identical cache keys.

    This is the core correctness requirement for prefix caching.
    If these tests fail, cache lookups will miss on equivalent prompts.
    """

    def test_whitespace_equivalence(self):
        """Prompts differing only in whitespace should match."""
        composer = PromptComposer()

        prompt1 = [Segment(SegmentType.SYSTEM, "Hello world")]
        prompt2 = [Segment(SegmentType.SYSTEM, "Hello  world")]
        prompt3 = [Segment(SegmentType.SYSTEM, "  Hello world  ")]

        result1 = composer.compose(prompt1)
        result2 = composer.compose(prompt2)
        result3 = composer.compose(prompt3)

        hash1 = composer.compute_prefix_hash(result1)
        hash2 = composer.compute_prefix_hash(result2)
        hash3 = composer.compute_prefix_hash(result3)

        assert hash1 == hash2 == hash3

    def test_json_equivalence(self):
        """Prompts with equivalent JSON should match."""
        composer = PromptComposer()

        prompt1 = [Segment(SegmentType.SYSTEM, 'Tools: {"a": 1, "b": 2}')]
        prompt2 = [Segment(SegmentType.SYSTEM, 'Tools: {"b":2,"a":1}')]

        result1 = composer.compose(prompt1)
        result2 = composer.compose(prompt2)

        hash1 = composer.compute_prefix_hash(result1)
        hash2 = composer.compute_prefix_hash(result2)

        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Semantically different prompts must have different hashes."""
        composer = PromptComposer()

        prompt1 = [Segment(SegmentType.SYSTEM, "Be helpful")]
        prompt2 = [Segment(SegmentType.SYSTEM, "Be harmful")]  # Different!

        result1 = composer.compose(prompt1)
        result2 = composer.compose(prompt2)

        hash1 = composer.compute_prefix_hash(result1)
        hash2 = composer.compute_prefix_hash(result2)

        assert hash1 != hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
