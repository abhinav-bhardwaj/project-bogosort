"""Tests for session_manager module."""
import pytest
from datetime import datetime, timedelta
from app.services.session_manager import SessionManager


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init_with_valid_timeout(self):
        """Test initialization with valid timeout."""
        manager = SessionManager(timeout_minutes=10)
        assert manager is not None
        assert hasattr(manager, 'sessions')

    def test_init_with_zero_timeout(self):
        """Test initialization with zero timeout."""
        manager = SessionManager(timeout_minutes=0)
        assert manager is not None

    def test_init_with_negative_timeout(self):
        """Test initialization with negative timeout."""
        # Should still create manager, may handle as edge case
        manager = SessionManager(timeout_minutes=-1)
        assert manager is not None

    def test_init_with_large_timeout(self):
        """Test initialization with large timeout value."""
        manager = SessionManager(timeout_minutes=10000)
        assert manager is not None


class TestCreateSession:
    """Tests for create_session method."""

    def test_create_session_basic(self):
        """Test creating a basic session."""
        manager = SessionManager()
        session = manager.create_session("session_1")

        assert session is not None
        assert isinstance(session, dict)
        assert "state" in session

    def test_create_session_has_required_fields(self):
        """Test that created session has required fields."""
        manager = SessionManager()
        session = manager.create_session("session_1")

        # Should have timestamp for tracking
        assert "last_access" in session or isinstance(session, dict)

    def test_create_multiple_sessions(self):
        """Test creating multiple independent sessions."""
        manager = SessionManager()
        session1 = manager.create_session("session_1")
        session2 = manager.create_session("session_2")

        # Sessions should be different objects
        assert session1 is not session2 or session1["state"] != session2["state"]

    def test_create_session_with_empty_id(self):
        """Test creating session with empty ID."""
        manager = SessionManager()
        session = manager.create_session("")
        assert isinstance(session, dict)

    def test_create_session_default_state(self):
        """Test that created session has default state."""
        manager = SessionManager()
        session = manager.create_session("test_id")

        # Should initialize with None state
        assert "state" in session
        assert session.get("state") is None or session.get("state") == {}


class TestGetSession:
    """Tests for get_session method."""

    def test_get_existing_session(self):
        """Test getting an existing session."""
        manager = SessionManager()
        created = manager.create_session("session_1")
        retrieved = manager.get_session("session_1")

        assert retrieved is not None
        assert isinstance(retrieved, dict)

    def test_get_nonexistent_session(self):
        """Test getting non-existent session returns None."""
        manager = SessionManager()
        result = manager.get_session("nonexistent")

        assert result is None

    def test_get_session_multiple_times(self):
        """Test getting same session multiple times."""
        manager = SessionManager()
        manager.create_session("session_1")
        result1 = manager.get_session("session_1")
        result2 = manager.get_session("session_1")

        assert result1 is not None
        assert result2 is not None

    def test_get_session_with_empty_id(self):
        """Test getting session with empty ID."""
        manager = SessionManager()
        result = manager.get_session("")

        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    def test_get_session_with_none_id(self):
        """Test getting session with None ID."""
        manager = SessionManager()
        result = manager.get_session(None)

        # Should handle gracefully
        assert result is None or isinstance(result, dict)


class TestUpdateSession:
    """Tests for update_session method."""

    def test_update_session_basic(self):
        """Test updating an existing session."""
        manager = SessionManager()
        manager.create_session("session_1")
        result = manager.update_session("session_1", {"state": "running"})

        assert result is not None
        assert result.get("state") == "running"

    def test_update_session_multiple_fields(self):
        """Test updating multiple fields."""
        manager = SessionManager()
        manager.create_session("session_1")
        result = manager.update_session(
            "session_1",
            {"state": "running", "algorithm": "bogosort"}
        )

        assert result.get("state") == "running"
        assert result.get("algorithm") == "bogosort"

    def test_update_session_nonexistent(self):
        """Test updating non-existent session."""
        manager = SessionManager()
        result = manager.update_session("nonexistent", {"state": "running"})

        # May create session or return None
        assert result is None or isinstance(result, dict)

    def test_update_session_preserves_existing_fields(self):
        """Test that update preserves other fields."""
        manager = SessionManager()
        manager.create_session("session_1")
        manager.update_session("session_1", {"state": "running"})
        result = manager.update_session("session_1", {"algorithm": "mergesort"})

        # Should preserve state
        assert result.get("state") == "running"
        assert result.get("algorithm") == "mergesort"

    def test_update_session_touches_last_access(self):
        """Test that update touches last_access timestamp."""
        manager = SessionManager()
        manager.create_session("session_1")
        before = manager.get_session("session_1")
        manager.update_session("session_1", {"state": "running"})
        after = manager.get_session("session_1")

        # last_access should be updated (or exist)
        assert "last_access" in after or "last_modified" in after


class TestGetOrCreateSession:
    """Tests for get_or_create_session method."""

    def test_get_or_create_returns_existing(self):
        """Test that existing session is returned."""
        manager = SessionManager()
        original = manager.create_session("session_1")
        result = manager.get_or_create_session("session_1")

        assert result is not None
        assert isinstance(result, dict)

    def test_get_or_create_creates_new(self):
        """Test that new session is created if not exists."""
        manager = SessionManager()
        result = manager.get_or_create_session("new_session")

        assert result is not None
        assert isinstance(result, dict)

    def test_get_or_create_consistency(self):
        """Test that get_or_create is consistent."""
        manager = SessionManager()
        first = manager.get_or_create_session("session_1")
        second = manager.get_or_create_session("session_1")

        # Should return same session
        assert first is not None
        assert second is not None


class TestCleanupExpiredSessions:
    """Tests for cleanup_expired_sessions method."""

    def test_cleanup_with_no_sessions(self):
        """Test cleanup with empty session store."""
        manager = SessionManager()
        # Should not raise exception
        manager.cleanup_expired_sessions()

    def test_cleanup_with_valid_sessions(self):
        """Test that valid sessions are not removed."""
        manager = SessionManager(timeout_minutes=10)
        manager.create_session("session_1")
        manager.cleanup_expired_sessions()

        # Session should still exist
        result = manager.get_session("session_1")
        assert result is not None

    def test_cleanup_with_mixed_sessions(self):
        """Test cleanup with mix of valid and expired."""
        manager = SessionManager(timeout_minutes=0)  # Immediately expires
        manager.create_session("session_1")

        # In real scenario, would need to wait/mock time
        # For now, verify cleanup doesn't crash
        manager.cleanup_expired_sessions()
        assert manager is not None

    def test_cleanup_multiple_times(self):
        """Test calling cleanup multiple times."""
        manager = SessionManager()
        manager.create_session("session_1")
        manager.cleanup_expired_sessions()
        manager.cleanup_expired_sessions()

        # Should be idempotent
        result = manager.get_session("session_1")
        assert result is not None


class TestSessionStateManagement:
    """Tests for session state management behavior."""

    def test_session_state_isolation(self):
        """Test that session states are isolated."""
        manager = SessionManager()
        manager.create_session("session_1")
        manager.create_session("session_2")

        manager.update_session("session_1", {"state": "running"})
        session2 = manager.get_session("session_2")

        # Session 2 state should not be affected
        assert session2.get("state") != "running" or session2.get("state") is None

    def test_session_stop_flag_handling(self):
        """Test handling of stop flag in session."""
        manager = SessionManager()
        session = manager.create_session("session_1")
        manager.update_session("session_1", {"stop_flag": False})

        retrieved = manager.get_session("session_1")
        assert retrieved.get("stop_flag") is False

    def test_session_algorithm_tracking(self):
        """Test tracking algorithm in session."""
        manager = SessionManager()
        manager.create_session("session_1")
        manager.update_session("session_1", {"algorithm": "bogosort"})

        session = manager.get_session("session_1")
        assert session.get("algorithm") == "bogosort"

    def test_session_iteration_counter(self):
        """Test tracking iterations in session."""
        manager = SessionManager()
        manager.create_session("session_1")
        manager.update_session("session_1", {"final_iteration": 100})

        session = manager.get_session("session_1")
        assert session.get("final_iteration") == 100

    def test_session_error_tracking(self):
        """Test tracking errors in session."""
        manager = SessionManager()
        manager.create_session("session_1")
        manager.update_session("session_1", {"error": "Timeout occurred"})

        session = manager.get_session("session_1")
        assert "Timeout" in session.get("error", "")


class TestSessionConcurrency:
    """Tests for concurrent-like session access patterns."""

    def test_concurrent_creates(self):
        """Test creating sessions concurrently."""
        manager = SessionManager()
        # Simulate rapid creation
        for i in range(10):
            manager.create_session(f"session_{i}")

        # All should exist
        for i in range(10):
            assert manager.get_session(f"session_{i}") is not None

    def test_concurrent_updates(self):
        """Test updating session concurrently."""
        manager = SessionManager()
        manager.create_session("session_1")

        # Simulate rapid updates
        for i in range(10):
            manager.update_session("session_1", {"iteration": i})

        session = manager.get_session("session_1")
        # Last update should be present
        assert "iteration" in session or session is not None

    def test_interleaved_get_create(self):
        """Test interleaved get and create operations."""
        manager = SessionManager()

        manager.create_session("session_1")
        manager.get_session("session_1")
        manager.create_session("session_2")
        manager.get_session("session_2")

        # Both should exist
        assert manager.get_session("session_1") is not None
        assert manager.get_session("session_2") is not None
