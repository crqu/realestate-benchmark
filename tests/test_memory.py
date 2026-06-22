"""Tests for the Memory system."""

from unittest.mock import Mock

import pytest

from realestate_benchmark.agents.memory import MEMORY_TEMPLATE, Memory


class TestMemory:
    """Test suite for Memory class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db = Mock()
        self.game_id = "test-game-123"
        self.agent_id = "seller"
        self.memory = Memory(self.agent_id, self.db, self.game_id)

    def test_initialization(self):
        """Test memory initializes with template."""
        assert self.memory.agent_id == self.agent_id
        assert self.memory.db == self.db
        assert self.memory.game_id == self.game_id
        assert self.memory.content == MEMORY_TEMPLATE
        assert self.memory.turn == 0

    def test_read(self):
        """Test reading memory content."""
        assert self.memory.read() == MEMORY_TEMPLATE

    def test_write(self):
        """Test replacing memory content."""
        new_content = "# New Memory\n\nThis is new content."
        self.memory.write(new_content)
        assert self.memory.read() == new_content

    def test_append(self):
        """Test appending to memory."""
        note = "- Important observation"
        initial_content = self.memory.content
        self.memory.append(note)

        content = self.memory.read()
        assert note in content
        assert content != initial_content
        # Should add newline if not present
        assert content.endswith("\n")

    def test_append_multiple(self):
        """Test multiple appends."""
        self.memory.append("- First note")
        self.memory.append("- Second note")

        content = self.memory.read()
        assert "- First note" in content
        assert "- Second note" in content

    def test_append_multiline(self):
        """Test appending multiline text."""
        note = "- First line\n- Second line"
        self.memory.append(note)

        content = self.memory.read()
        assert "- First line" in content
        assert "- Second line" in content

    def test_snapshot(self):
        """Test saving snapshot to database."""
        turn = 5
        self.memory.append("- Turn 5 observation")

        self.memory.snapshot(turn)

        # Verify database was called correctly
        self.db.save_memory_snapshot.assert_called_once_with(
            game_id=self.game_id,
            agent_id=self.agent_id,
            turn=turn,
            content=self.memory.content,
        )
        assert self.memory.turn == turn

    def test_load_snapshot(self):
        """Test loading historical snapshot."""
        turn = 3
        historical_content = "# Historical Memory\n\n- Old observation"
        self.db.load_memory_snapshot.return_value = historical_content

        loaded = self.memory.load_snapshot(turn)

        assert loaded == historical_content
        self.db.load_memory_snapshot.assert_called_once_with(
            game_id=self.game_id,
            agent_id=self.agent_id,
            turn=turn,
        )

    def test_memory_template_structure(self):
        """Test that template has expected sections."""
        content = self.memory.read()
        assert "# Agent Memory" in content
        assert "## Facts" in content
        assert "## Observations" in content
        assert "## Strategy" in content

    def test_memory_workflow(self):
        """Test typical memory usage workflow."""
        # Initialize memory
        memory = Memory("buyer", self.db, "game-456")

        # Add some facts
        memory.append("## Facts")
        memory.append("- Budget: $200,000")
        memory.append("- Looking for 3-bedroom home")

        # Snapshot at turn 1
        memory.snapshot(1)
        assert self.db.save_memory_snapshot.call_count == 1

        # Add observations
        memory.append("## Observations")
        memory.append("- Seller seems hesitant about basement questions")

        # Snapshot at turn 3
        memory.snapshot(3)
        assert self.db.save_memory_snapshot.call_count == 2

        # Verify content includes all additions
        content = memory.read()
        assert "Budget: $200,000" in content
        assert "Seller seems hesitant" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
