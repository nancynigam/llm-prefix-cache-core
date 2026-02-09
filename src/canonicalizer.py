"""
Text canonicalization for stable tokenization.

INTERVIEW KEY CONCEPT:
Tokenizers are sensitive to subtle text differences. For example:
    "Hello world" -> [15496, 1917]
    "Hello  world" -> [15496, 220, 1917]  # Extra space = different tokens!

If we don't canonicalize, two "equivalent" prompts produce different tokens,
and our cache lookup fails even though the content is logically identical.

The canonicalizer normalizes text so that equivalent inputs always produce
identical token sequences. This is a correctness requirement, not an optimization.
"""

import json
import re
import unicodedata
from typing import Optional

from .config import CanonicalizeConfig


class Canonicalizer:
    """
    Normalizes text for deterministic tokenization.

    Why each normalization matters:
    1. Whitespace: "a  b" vs "a b" tokenize differently
    2. Unicode: "é" (composed) vs "é" (decomposed) tokenize differently
    3. JSON: {"a":1,"b":2} vs {"b": 2, "a": 1} are semantically equal
    """

    def __init__(self, config: Optional[CanonicalizeConfig] = None):
        self.config = config or CanonicalizeConfig()

        # Regex for collapsing multiple whitespace to single space
        self._whitespace_re = re.compile(r'[ \t]+')

        # Regex for normalizing newlines
        self._newline_re = re.compile(r'\n{3,}')

    def canonicalize(self, text: str) -> str:
        """
        Apply all configured normalizations to text.

        Order matters! We apply normalizations in a specific sequence
        to ensure consistent results.
        """
        if not text:
            return text

        result = text

        # 1. Unicode normalization (NFC = composed form)
        # This must come first because it affects character boundaries
        if self.config.normalize_unicode:
            result = self._normalize_unicode(result)

        # 2. JSON normalization (before whitespace, as it adds consistent spacing)
        if self.config.normalize_json:
            result = self._normalize_json_in_text(result)

        # 3. Whitespace normalization
        if self.config.normalize_whitespace:
            result = self._normalize_whitespace(result)

        # 4. Strip (after other normalizations)
        if self.config.strip_segments:
            result = result.strip()

        return result

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode to NFC (Canonical Decomposition, followed by Canonical Composition).

        Why NFC?
        - "é" can be represented as single char (U+00E9) or as "e" + combining accent
        - NFC ensures we always use the composed single-character form
        - This is what most systems expect and produce
        """
        return unicodedata.normalize('NFC', text)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace while preserving structure.

        Rules:
        - Collapse multiple spaces/tabs to single space
        - Preserve single newlines (they often have semantic meaning)
        - Collapse 3+ consecutive newlines to 2
        - Don't strip (that's handled separately)
        """
        # Collapse horizontal whitespace (spaces, tabs) to single space
        result = self._whitespace_re.sub(' ', text)

        # Collapse excessive newlines (3+ -> 2)
        result = self._newline_re.sub('\n\n', result)

        return result

    def _normalize_json_in_text(self, text: str) -> str:
        """
        Find and normalize JSON objects/arrays embedded in text.

        Why normalize JSON?
        - {"a": 1, "b": 2} vs {"b": 2, "a": 1} are semantically identical
        - But they tokenize differently
        - This is common in tool definitions, API specs, etc.

        We detect JSON by looking for {...} or [...] patterns and attempting
        to parse them. If parsing succeeds, we re-serialize with sorted keys
        and consistent formatting.
        """
        # Find potential JSON objects/arrays
        # This regex finds balanced braces/brackets (simplified - doesn't handle all edge cases)
        result = text

        # Try to find and normalize JSON objects
        result = self._try_normalize_json_objects(result)

        return result

    def _try_normalize_json_objects(self, text: str) -> str:
        """
        Attempt to find and normalize JSON in text.

        Strategy: Look for substrings that start with { or [ and try to parse them.
        This is heuristic but handles common cases well.
        """
        result = []
        i = 0

        while i < len(text):
            if text[i] in '{[':
                # Try to find the end of this JSON structure
                json_str, end_pos = self._extract_json(text, i)
                if json_str is not None:
                    # Successfully parsed - normalize and add
                    result.append(json_str)
                    i = end_pos
                    continue

            result.append(text[i])
            i += 1

        return ''.join(result)

    def _extract_json(self, text: str, start: int) -> tuple[Optional[str], int]:
        """
        Try to extract and normalize JSON starting at position start.

        Returns (normalized_json, end_position) or (None, start) if not valid JSON.
        """
        # Find potential end by counting brackets
        open_char = text[start]
        close_char = '}' if open_char == '{' else ']'

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    # Found the end - try to parse
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        # Normalize: sorted keys, compact separators
                        normalized = json.dumps(
                            parsed,
                            sort_keys=True,
                            separators=(',', ':'),  # No extra spaces
                            ensure_ascii=False
                        )
                        return normalized, i + 1
                    except json.JSONDecodeError:
                        return None, start

        return None, start


def canonicalize_text(text: str, config: Optional[CanonicalizeConfig] = None) -> str:
    """
    Convenience function for one-off canonicalization.

    Usage:
        canonical = canonicalize_text("  Hello   world  ")
        # Returns: "Hello world"
    """
    return Canonicalizer(config).canonicalize(text)
