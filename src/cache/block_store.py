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


class KVBlockStore:
    """
    Manages allocation of blocks for token sequences.

    Responsibilities (for now):
    - Split token sequences into fixed-size blocks
    - Assign unique block IDs
    - Track total memory usage

    Example:
        store = KVBlockStore(block_size=64)
        blocks = store.allocate(tokens=[1, 2, 3, ... 150 tokens])
        # Returns 3 blocks: [64 tokens], [64 tokens], [22 tokens]
    """

    def __init__(self, block_size: int = 64):
        """
        Args:
            block_size: Number of tokens per block (default 64)
        """
        self.block_size = block_size
        self._next_block_id = 0
        self._blocks: dict[int, Block] = {}  # block_id -> Block

    def allocate(self, tokens: list[int]) -> list[Block]:
        """
        Split tokens into blocks and return them.

        Args:
            tokens: List of token IDs to store

        Returns:
            List of Block objects containing the tokens
        """
        if not tokens:
            return []

        blocks = []

        # Split tokens into chunks of block_size
        for i in range(0, len(tokens), self.block_size):
            chunk = tokens[i : i + self.block_size]

            block = Block(
                block_id=self._next_block_id,
                tokens=chunk,
            )

            self._blocks[block.block_id] = block
            self._next_block_id += 1
            blocks.append(block)

        return blocks

    def get_block(self, block_id: int) -> Optional[Block]:
        """Get a block by its ID."""
        return self._blocks.get(block_id)

    def free_block(self, block_id: int) -> bool:
        """
        Remove a block from the store.
        Only succeeds if block exists and ref_count is 0.

        Returns True if freed, False otherwise.
        """
        block = self._blocks.get(block_id)
        if block is None:
            return False

        if not block.is_free:
            return False

        del self._blocks[block_id]
        return True

    @property
    def num_blocks(self) -> int:
        """Total number of blocks in store."""
        return len(self._blocks)

    @property
    def total_memory_bytes(self) -> int:
        """Total memory used by all blocks."""
        return sum(block.size_bytes for block in self._blocks.values())
