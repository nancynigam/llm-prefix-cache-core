"""
Block and KVBlockStore for managing KV cache storage.

KEY CONCEPT: Blocks
Instead of storing whole prefixes, we split them into fixed-size blocks.
This enables sharing: if two prefixes share a common start, they share blocks.

Example:
    Prefix A: "You are helpful. Be concise." → [Block 1][Block 2][Block 3]
    Prefix B: "You are helpful. Be detailed." → [Block 1][Block 2][Block 4]
                                                     ↑ Shared blocks
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Block:
    """
    A fixed-size container for tokens and their KV cache data.

    Why fixed-size blocks?
    1. Memory efficiency: Shared prefixes share blocks
    2. Simple allocation: No fragmentation issues
    3. Matches vLLM's paged attention design

    Attributes:
        block_id: Unique identifier for this block
        tokens: List of token IDs stored in this block (max block_size)
        ref_count: Number of cache entries using this block
        last_access: Timestamp of last access (for LRU eviction)
        kv_data: The actual KV tensors (None until populated by model)
    """
    block_id: int
    tokens: list[int]
    ref_count: int = 0
    last_access: float = field(default_factory=time.time)
    kv_data: Optional[Any] = None  # Will be torch.Tensor when ML libs installed

    def touch(self) -> None:
        """Update last_access timestamp (called on cache hit)."""
        self.last_access = time.time()

    def incr_ref(self) -> None:
        """Increment reference count (called when a cache entry uses this block)."""
        self.ref_count += 1
        self.touch()

    def decr_ref(self) -> int:
        """
        Decrement reference count (called when a cache entry is evicted).
        Returns the new ref_count.
        """
        self.ref_count = max(0, self.ref_count - 1)
        return self.ref_count

    @property
    def is_free(self) -> bool:
        """Block can be freed when no one references it."""
        return self.ref_count == 0

    @property
    def size_bytes(self) -> int:
        """
        Estimate memory usage of this block.

        For now, we estimate based on tokens only.
        When kv_data is populated, we'd include tensor size.
        """
        # Base: token IDs (8 bytes each for Python int)
        token_bytes = len(self.tokens) * 8

        # KV data if present (torch tensor)
        kv_bytes = 0
        if self.kv_data is not None:
            # Check if it's a torch tensor with numel/element_size methods
            if hasattr(self.kv_data, 'numel') and hasattr(self.kv_data, 'element_size'):
                kv_bytes = self.kv_data.numel() * self.kv_data.element_size()

        return token_bytes + kv_bytes

    def __repr__(self) -> str:
        return (
            f"Block(id={self.block_id}, "
            f"tokens={len(self.tokens)}, "
            f"ref_count={self.ref_count})"
        )
