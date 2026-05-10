"""Tests for bogosort routes and helper functions."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.routes.bogosort import (
    is_sorted,
    load_shuffled_toxic_words,
    bogosort_snapshots,
    bogosort,
    background_bogosort
)


class TestIsSorted:
    """Test is_sorted helper function."""

    def test_sorted_array(self):
        """Test that sorted array is recognized."""
        assert is_sorted([5, 4, 3, 2, 1]) is True

    def test_sorted_equal_values(self):
        """Test array with equal consecutive values."""
        assert is_sorted([5, 5, 3, 3, 1]) is True

    def test_unsorted_array(self):
        """Test that unsorted array is detected."""
        assert is_sorted([1, 5, 3, 2, 4]) is False

    def test_single_element(self):
        """Test single element array."""
        assert is_sorted([1]) is True

    def test_two_elements_sorted(self):
        """Test two sorted elements."""
        assert is_sorted([5, 3]) is True

    def test_two_elements_unsorted(self):
        """Test two unsorted elements."""
        assert is_sorted([2, 5]) is False

    def test_empty_array(self):
        """Test empty array."""
        assert is_sorted([]) is True


class TestLoadShuffledToxicWords:
    """Test load_shuffled_toxic_words function."""

    @patch('app.routes.bogosort.np.load')
    def test_load_and_shuffle_data(self, mock_load, mock_toxic_words):
        """Test loading and shuffling toxic words."""
        mock_load.return_value = np.array(mock_toxic_words, dtype=object)

        words, counts = load_shuffled_toxic_words()

        assert len(words) == len(mock_toxic_words)
        assert len(counts) == len(mock_toxic_words)
        assert all(isinstance(w, str) for w in words)
        assert all(isinstance(c, (int, np.integer)) for c in counts)

    @patch('app.routes.bogosort.np.load')
    def test_counts_are_integers(self, mock_load, mock_toxic_words):
        """Test that counts are converted to integers."""
        mock_load.return_value = np.array(mock_toxic_words, dtype=object)

        words, counts = load_shuffled_toxic_words()

        assert all(isinstance(c, (int, np.integer)) for c in counts)

    @patch('app.routes.bogosort.np.load')
    def test_data_is_unpacked_correctly(self, mock_load):
        """Test that tuples are unpacked into words and counts."""
        data = [("word1", 10), ("word2", 20), ("word3", 30)]
        mock_load.return_value = np.array(data, dtype=object)

        words, counts = load_shuffled_toxic_words()

        assert set(words) == {"word1", "word2", "word3"}
        assert set(counts) == {10, 20, 30}


class TestBogosortSnapshots:
    """Test bogosort_snapshots function."""

    def test_snapshots_generation(self, mock_toxic_words):
        """Test generating bogosort snapshots."""
        words, counts = zip(*mock_toxic_words)
        snapshots = bogosort_snapshots(list(words), list(counts), max_iterations=100)

        assert len(snapshots) > 0
        # Each snapshot should have a state array and iteration number
        for snap, iteration in snapshots:
            assert len(snap) == len(mock_toxic_words)
            assert isinstance(iteration, int)

    def test_snapshots_have_iteration_count(self, mock_toxic_words):
        """Test that each snapshot includes iteration count."""
        words, counts = zip(*mock_toxic_words)
        snapshots = bogosort_snapshots(list(words), list(counts), max_iterations=100)

        for snap, iteration in snapshots:
            assert isinstance(iteration, int)
            assert iteration >= 0

    def test_snapshots_respects_max_iterations(self, mock_toxic_words):
        """Test that max_iterations limit is respected."""
        words, counts = zip(*mock_toxic_words)
        max_iter = 50
        snapshots = bogosort_snapshots(list(words), list(counts), max_iterations=max_iter)

        # The last snapshot should be at or before max_iterations
        assert snapshots[-1][1] <= max_iter

    def test_snapshots_last_is_final_state(self, mock_toxic_words):
        """Test that last snapshot is the final state."""
        words, counts = zip(*mock_toxic_words)
        snapshots = bogosort_snapshots(list(words), list(counts), max_iterations=100)

        last_snap, last_iter = snapshots[-1]
        assert len(last_snap) == len(mock_toxic_words)


class TestBogosortRoute:
    """Test bogosort route handler."""

    @patch('app.routes.bogosort.load_shuffled_toxic_words')
    @patch('app.routes.bogosort.save_distribution_plot')
    def test_bogosort_get_idle_state(self, mock_save_plot, mock_load_words, client, mock_toxic_words):
        """Test bogosort GET request in idle state."""
        mock_load_words.return_value = zip(*mock_toxic_words)

        response = client.get("/bogosort/")
        assert response.status_code == 200

    @patch('app.routes.bogosort.load_shuffled_toxic_words')
    @patch('app.routes.bogosort.save_distribution_plot')
    def test_bogosort_post_starts_sorting(self, mock_save_plot, mock_load_words, client, mock_toxic_words):
        """Test that POST request starts background sorting."""
        mock_load_words.return_value = zip(*mock_toxic_words)

        response = client.post("/bogosort/")
        assert response.status_code == 302  # Redirect

    @patch('app.routes.bogosort.load_shuffled_toxic_words')
    @patch('app.routes.bogosort.save_distribution_plot')
    def test_bogosort_route_exists(self, mock_save_plot, mock_load_words, client, mock_toxic_words):
        """Test that bogosort route is accessible."""
        mock_load_words.return_value = zip(*mock_toxic_words)

        response = client.get("/bogosort/")
        assert response.status_code == 200


class TestBackgroundBogosort:
    """Test background_bogosort function."""

    @patch('app.routes.bogosort.bogosort_snapshots')
    @patch('app.routes.bogosort.save_sort_animation')
    def test_background_sort_success(self, mock_save_anim, mock_snapshots):
        """Test successful background sorting."""
        mock_snapshots.return_value = [
            ([("toxic", 150), ("hate", 120), ("abuse", 100), ("spam", 80), ("rude", 60)], 0),
            ([("rude", 150), ("spam", 120), ("abuse", 100), ("hate", 80), ("toxic", 60)], 100),
        ]

        status = {"state": None}
        background_bogosort(["toxic", "hate"], [150, 120], "test.gif", status)

        assert status["state"] == "done"
        assert mock_save_anim.called

    @patch('app.routes.bogosort.bogosort_snapshots')
    @patch('app.routes.bogosort.save_sort_animation')
    def test_background_sort_error_handling(self, mock_save_anim, mock_snapshots):
        """Test error handling in background sorting."""
        mock_snapshots.side_effect = Exception("Test error")

        status = {"state": None}
        background_bogosort(["toxic"], [150], "test.gif", status)

        assert status["state"] == "error"
        assert status["err"] == "Test error"

    @patch('app.routes.bogosort.bogosort_snapshots')
    @patch('app.routes.bogosort.save_sort_animation')
    def test_background_sort_sets_iteration_count(self, mock_save_anim, mock_snapshots):
        """Test that background sort sets final iteration count."""
        mock_snapshots.return_value = [
            ([("a", 1)], 0),
            ([("a", 1)], 50),
        ]

        status = {"state": None}
        background_bogosort(["a"], [1], "test.gif", status)

        assert status["final_iteration"] == 50
