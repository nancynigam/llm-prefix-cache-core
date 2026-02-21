"""
Tests for Block and KVBlockStore.

Run with: pytest tests/test_block_store.py -v
"""

import pytest
import time

from src.cache.block_store import Block, KVBlockStore


class TestBlock:
    """Tests for the Block dataclass."""

    def test_block_creation(self):
        """Can create a block with tokens."""
        block = Block(block_id=1, tokens=[100, 200, 300])

        assert block.block_id == 1
        assert block.tokens == [100, 200, 300]
        assert block.ref_count == 0
        assert block.kv_data is None

    def test_ref_count_increment(self):
        """incr_ref increases ref_count."""
        block = Block(block_id=1, tokens=[1, 2, 3])

        assert block.ref_count == 0
        block.incr_ref()
        assert block.ref_count == 1
        block.incr_ref()
        assert block.ref_count == 2

    def test_ref_count_decrement(self):
        """decr_ref decreases ref_count, never goes below 0."""
        block = Block(block_id=1, tokens=[1, 2, 3])
        block.ref_count = 2

        assert block.decr_ref() == 1
        assert block.decr_ref() == 0
        assert block.decr_ref() == 0  # Doesn't go negative

    def test_is_free(self):
        """is_free returns True when ref_count is 0."""
        block = Block(block_id=1, tokens=[1, 2, 3])

        assert block.is_free is True

        block.incr_ref()
        assert block.is_free is False

        block.decr_ref()
        assert block.is_free is True

    def test_touch_updates_timestamp(self):
        """touch() updates last_access."""
        block = Block(block_id=1, tokens=[1, 2, 3])
        old_time = block.last_access

        time.sleep(0.01)  # Small delay
        block.touch()

        assert block.last_access > old_time

    def test_size_bytes_tokens_only(self):
        """size_bytes estimates memory for tokens."""
        block = Block(block_id=1, tokens=[1, 2, 3, 4])

        # 4 tokens × 8 bytes each = 32 bytes
        assert block.size_bytes == 32

    def test_repr(self):
        """Block has readable repr."""
        block = Block(block_id=42, tokens=[1, 2, 3])
        block.incr_ref()

        repr_str = repr(block)
        assert "42" in repr_str
        assert "3" in repr_str  # 3 tokens
        assert "ref_count=1" in repr_str


class TestKVBlockStore:
    """Tests for KVBlockStore allocation."""

    def test_allocate_single_block(self):
        """Tokens fitting in one block."""
        store = KVBlockStore(block_size=64)
        tokens = list(range(50))  # 50 tokens < 64

        blocks = store.allocate(tokens)

        assert len(blocks) == 1
        assert blocks[0].tokens == tokens
        assert store.num_blocks == 1

    def test_allocate_multiple_blocks(self):
        """Tokens split across multiple blocks."""
        store = KVBlockStore(block_size=64)
        tokens = list(range(150))  # 150 tokens = 3 blocks

        blocks = store.allocate(tokens)

        assert len(blocks) == 3
        assert len(blocks[0].tokens) == 64
        assert len(blocks[1].tokens) == 64
        assert len(blocks[2].tokens) == 22  # Remainder
        assert store.num_blocks == 3

    def test_allocate_exact_block_size(self):
        """Tokens exactly filling blocks."""
        store = KVBlockStore(block_size=64)
        tokens = list(range(128))  # Exactly 2 blocks

        blocks = store.allocate(tokens)

        assert len(blocks) == 2
        assert len(blocks[0].tokens) == 64
        assert len(blocks[1].tokens) == 64

    def test_allocate_empty(self):
        """Empty token list returns no blocks."""
        store = KVBlockStore(block_size=64)

        blocks = store.allocate([])

        assert blocks == []
        assert store.num_blocks == 0

    def test_unique_block_ids(self):
        """Each allocation gets unique block IDs."""
        store = KVBlockStore(block_size=64)

        blocks1 = store.allocate(list(range(100)))
        blocks2 = store.allocate(list(range(100)))

        ids1 = [b.block_id for b in blocks1]
        ids2 = [b.block_id for b in blocks2]

        # No overlap
        assert set(ids1).isdisjoint(set(ids2))

    def test_get_block(self):
        """Can retrieve block by ID."""
        store = KVBlockStore(block_size=64)
        blocks = store.allocate([1, 2, 3])

        retrieved = store.get_block(blocks[0].block_id)

        assert retrieved is blocks[0]

    def test_get_block_not_found(self):
        """Returns None for unknown block ID."""
        store = KVBlockStore(block_size=64)

        assert store.get_block(999) is None

    def test_free_block(self):
        """Can free a block with ref_count=0."""
        store = KVBlockStore(block_size=64)
        blocks = store.allocate([1, 2, 3])
        block_id = blocks[0].block_id

        assert store.num_blocks == 1
        assert store.free_block(block_id) is True
        assert store.num_blocks == 0

    def test_free_block_with_refs(self):
        """Cannot free a block with ref_count > 0."""
        store = KVBlockStore(block_size=64)
        blocks = store.allocate([1, 2, 3])
        blocks[0].incr_ref()  # Someone is using it

        assert store.free_block(blocks[0].block_id) is False
        assert store.num_blocks == 1  # Still there

    def test_total_memory_bytes(self):
        """Tracks total memory across blocks."""
        store = KVBlockStore(block_size=64)
        store.allocate([1, 2, 3, 4])  # 4 tokens = 32 bytes

        assert store.total_memory_bytes == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
