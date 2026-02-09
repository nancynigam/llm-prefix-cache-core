"""
LLM Prefix Cache - Production-safe KV prefix caching for LLM inference.

This package provides components for caching KV (Key-Value) projections
from transformer attention layers, enabling faster Time-To-First-Token (TTFT)
by skipping redundant prefill computation for shared prompt prefixes.

Main components:
- PromptComposer: Assembles prompt segments with cache-aware boundaries
- Canonicalizer: Normalizes text for deterministic tokenization
- PrefixCache: (Week 1) Exact-match prefix cache with LRU eviction
- KVBlockStore: (Week 1) Block-based KV storage with refcounting
"""

from .canonicalizer import Canonicalizer, canonicalize_text
from .composer import PromptComposer, create_simple_prompt
from .config import (
    CacheConfig,
    CanonicalizeConfig,
    ComposedPrompt,
    Segment,
    SegmentType,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "CacheConfig",
    "CanonicalizeConfig",
    "ComposedPrompt",
    "Segment",
    "SegmentType",
    # Canonicalizer
    "Canonicalizer",
    "canonicalize_text",
    # Composer
    "PromptComposer",
    "create_simple_prompt",
]
