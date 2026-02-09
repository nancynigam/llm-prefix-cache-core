"""
Configuration classes for the LLM Prefix Cache.

Interview note: Using dataclasses for configuration makes the system
inspectable and testable. Each config option is explicit and typed.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SegmentType(Enum):
    """
    Types of prompt segments with different caching behaviors.

    Why segments matter:
    - SYSTEM prompts are highly cacheable (same across many requests)
    - TEMPLATE prompts (few-shot examples) have medium reuse
    - USER prompts are unique per request (rarely cached)

    This hierarchy lets us cache at the right boundaries.
    """
    SYSTEM = "system"      # System instructions - highest reuse
    TEMPLATE = "template"  # Few-shot examples, tools - medium reuse
    USER = "user"          # User query - lowest reuse


@dataclass
class CacheConfig:
    """
    Configuration for the prefix cache.

    Interview note: These parameters control the memory/hit-rate tradeoff.
    Larger cache = more hits but more memory. Smaller blocks = finer sharing
    but more metadata overhead.
    """
    # Maximum memory for KV cache in bytes (default 1GB)
    max_memory_bytes: int = 1024 * 1024 * 1024

    # Tokens per block - balances fragmentation vs sharing granularity
    # 64 chosen based on typical system prompt lengths
    block_size_tokens: int = 64

    # Enable tenant isolation (required for multi-tenant production)
    enable_tenant_isolation: bool = True

    # Eviction policy: "lru" (Week 1) or "value_based" (Week 3)
    eviction_policy: str = "lru"

    # Maximum entries in cache (0 = unlimited, rely on memory cap)
    max_entries: int = 0


@dataclass
class CanonicalizeConfig:
    """
    Configuration for text canonicalization.

    Why canonicalize?
    Tokenizers are sensitive to whitespace and formatting. Two "equivalent"
    prompts with different whitespace produce different tokens, breaking
    cache lookups. Canonicalization ensures equivalent prompts -> identical tokens.
    """
    # Normalize whitespace (collapse multiple spaces, trim)
    normalize_whitespace: bool = True

    # Normalize JSON (consistent key ordering, no extra whitespace)
    normalize_json: bool = True

    # Strip leading/trailing whitespace from segments
    strip_segments: bool = True

    # Normalize unicode (NFC normalization)
    normalize_unicode: bool = True


@dataclass
class Segment:
    """
    A segment of a prompt with type and content.

    Segments are the building blocks of prompts. By marking boundaries,
    we know exactly where to cut for caching.

    Example:
        Segment(SegmentType.SYSTEM, "You are a helpful assistant.")
        Segment(SegmentType.USER, "What is 2+2?")
    """
    segment_type: SegmentType
    content: str

    # Optional metadata for cache key computation
    # e.g., {"tools_version": "v2", "policy_id": "default"}
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.segment_type, SegmentType):
            self.segment_type = SegmentType(self.segment_type)


@dataclass
class ComposedPrompt:
    """
    Result of composing segments into a prompt.

    Contains both the final text and metadata about segment boundaries,
    which we need for cache key computation and prefix identification.
    """
    # The full composed text
    text: str

    # Segment boundaries as character offsets: [(start, end, type), ...]
    # This lets us identify cacheable prefixes
    segment_boundaries: list[tuple[int, int, SegmentType]] = field(default_factory=list)

    # The original segments (for debugging/introspection)
    segments: list[Segment] = field(default_factory=list)

    @property
    def cacheable_prefix_end(self) -> int:
        """
        Returns character offset where cacheable prefix ends.

        We cache SYSTEM + TEMPLATE segments. USER content varies per request.
        """
        cacheable_end = 0
        for start, end, seg_type in self.segment_boundaries:
            if seg_type in (SegmentType.SYSTEM, SegmentType.TEMPLATE):
                cacheable_end = end
        return cacheable_end

    @property
    def cacheable_prefix(self) -> str:
        """Returns the cacheable portion of the prompt."""
        return self.text[:self.cacheable_prefix_end]
