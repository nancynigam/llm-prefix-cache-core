"""
PromptComposer - Assembles prompt segments with cache-aware boundaries.

KEY CONCEPT:
In production LLM serving, prompts typically have structure:
    [System Instructions] + [Few-shot Examples] + [User Query]

Each part has different caching characteristics:
- System: Same across all requests for an app -> highly cacheable
- Examples: Same for similar request types -> moderately cacheable
- User: Unique per request -> not cacheable

The PromptComposer:
1. Accepts typed segments
2. Canonicalizes SYSTEM and TEMPLATE segments for stable tokenization
   (USER segments are left as-is to preserve code, formatting, etc.)
3. Tracks boundaries so we know where to cut for caching
4. Produces a ComposedPrompt with all metadata needed for cache lookup

Why this matters for TTFT (Time to First Token):
If we cache [System + Examples] KV, we skip their prefill computation.
For a 2K token system prompt at ~50ms/1K tokens, that's ~100ms saved per request.
"""

import hashlib
from typing import Optional

from .canonicalizer import Canonicalizer, CanonicalizeConfig
from .config import (
    CacheConfig,
    ComposedPrompt,
    Segment,
    SegmentType,
)


class PromptComposer:
    """
    Composes prompt segments into a cacheable prompt structure.

    Usage:
        composer = PromptComposer()

        composed = composer.compose([
            Segment(SegmentType.SYSTEM, "You are a helpful assistant."),
            Segment(SegmentType.TEMPLATE, "Example: Q: Hi A: Hello!"),
            Segment(SegmentType.USER, "What is 2+2?"),
        ])

        print(composed.text)  # Full prompt
        print(composed.cacheable_prefix)  # Just system + template
    """

    # Separator between segments - chosen to be distinct but not waste tokens
    SEGMENT_SEPARATOR = "\n\n"

    def __init__(
        self,
        canonicalize_config: Optional[CanonicalizeConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        self.canonicalizer = Canonicalizer(canonicalize_config)
        self.cache_config = cache_config or CacheConfig()

    def compose(self, segments: list[Segment]) -> ComposedPrompt:
        """
        Compose segments into a single prompt with boundary tracking.

        Steps:
        1. Validate segment ordering (SYSTEM -> TEMPLATE -> USER)
        2. Canonicalize system & template segment
        3. Join with separators
        4. Track character offsets for each segment

        Returns ComposedPrompt with full text and metadata.
        """
        if not segments:
            return ComposedPrompt(text="", segment_boundaries=[], segments=[])

        # Validate ordering
        self._validate_segment_order(segments)

        # Build the composed text and track boundaries
        composed_parts = []
        boundaries = []
        current_offset = 0

        for i, segment in enumerate(segments):
            # Only canonicalize cacheable segments (SYSTEM, TEMPLATE)
            # USER content is left as-is to preserve code, formatting, etc.
            if segment.segment_type in (SegmentType.SYSTEM, SegmentType.TEMPLATE):
                content = self.canonicalizer.canonicalize(segment.content)
            else:
                content = segment.content

            # Add separator if not first segment
            if i > 0:
                composed_parts.append(self.SEGMENT_SEPARATOR)
                current_offset += len(self.SEGMENT_SEPARATOR)

            # Record boundary
            start = current_offset
            end = start + len(content)
            boundaries.append((start, end, segment.segment_type))

            # Add content
            composed_parts.append(content)
            current_offset = end

        full_text = ''.join(composed_parts)

        return ComposedPrompt(
            text=full_text,
            segment_boundaries=boundaries,
            segments=segments,
        )

    def _validate_segment_order(self, segments: list[Segment]) -> None:
        """
        Validate that segments appear in correct order.

        Required order: SYSTEM* -> TEMPLATE* -> USER*
        (Each type can appear 0+ times, but must be in order)

        Why enforce order?
        1. Consistent cache keys - same logical prompt = same key
        2. Correctness - system instructions should come first
        3. Debuggability - easier to reason about prompt structure
        """
        type_order = {
            SegmentType.SYSTEM: 0,
            SegmentType.TEMPLATE: 1,
            SegmentType.USER: 2,
        }

        current_max_order = -1
        for segment in segments:
            order = type_order[segment.segment_type]
            if order < current_max_order:
                raise ValueError(
                    f"Invalid segment order: {segment.segment_type.value} cannot appear "
                    f"after a segment of higher precedence. "
                    f"Required order: SYSTEM -> TEMPLATE -> USER"
                )
            current_max_order = max(current_max_order, order)

    def compute_prefix_hash(
        self,
        composed: ComposedPrompt,
        tenant_id: str = "default",
        model_id: str = "default",
    ) -> str:
        """
        Compute a hash of the cacheable prefix for cache lookup.

        The hash includes:
        - tenant_id: For multi-tenant isolation
        - model_id: Different models have different KV caches
        - prefix text: The actual cacheable content
        - segment metadata: Tools, policies, etc.

        Note: This is the cache key. It must be:
        1. Deterministic (same inputs -> same hash)
        2. Collision-resistant (different inputs -> different hash)
        3. Include all factors that affect KV cache validity
        """
        # Gather all metadata from cacheable segments
        metadata_parts = []
        for segment in composed.segments:
            if segment.segment_type in (SegmentType.SYSTEM, SegmentType.TEMPLATE):
                if segment.metadata:
                    # Sort keys for determinism
                    sorted_meta = sorted(segment.metadata.items())
                    metadata_parts.append(str(sorted_meta))

        # Build the hash input
        hash_input = "|".join([
            tenant_id,
            model_id,
            composed.cacheable_prefix,
            ":".join(metadata_parts),
        ])

        # SHA-256 for collision resistance
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    def compute_token_boundary(
        self,
        composed: ComposedPrompt,
        tokenizer,  # HuggingFace tokenizer
    ) -> int:
        """
        Compute the token offset where cacheable prefix ends.

        Why token boundary matters:
        We cache KV at token level, not character level. We need to know
        exactly which token position marks the end of the cacheable prefix.

        Note: This is approximate. Token boundaries don't always align with
        character boundaries. We round down to ensure we only cache what's
        fully in the prefix.
        """
        # Tokenize the full text
        full_tokens = tokenizer.encode(composed.text)

        # Tokenize just the prefix
        prefix_tokens = tokenizer.encode(composed.cacheable_prefix)

        # Verify prefix tokens are a prefix of full tokens
        # (They should be, but tokenizers can have edge cases)
        if full_tokens[:len(prefix_tokens)] != prefix_tokens:
            # Tokenization boundary mismatch - be conservative
            # Find the longest matching prefix
            match_len = 0
            for i in range(min(len(prefix_tokens), len(full_tokens))):
                if prefix_tokens[i] == full_tokens[i]:
                    match_len = i + 1
                else:
                    break
            return match_len

        return len(prefix_tokens)


def create_simple_prompt(
    system: str,
    user: str,
    template: Optional[str] = None,
) -> ComposedPrompt:
    """
    Convenience function for common prompt structure.

    Usage:
        prompt = create_simple_prompt(
            system="You are helpful.",
            user="What is 2+2?",
        )
    """
    composer = PromptComposer()
    segments = [Segment(SegmentType.SYSTEM, system)]

    if template:
        segments.append(Segment(SegmentType.TEMPLATE, template))

    segments.append(Segment(SegmentType.USER, user))

    return composer.compose(segments)
