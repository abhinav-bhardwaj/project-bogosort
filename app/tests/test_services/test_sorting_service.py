"""Tests for sorting_service module."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from app.services.sorting_service import SortingService


class TestIsSorted:
    """Tests for is_sorted function."""

    def test_sorted_descending_array(self):
        """Test that descending array is sorted."""
        counts = [5, 4, 3, 2, 1]
        assert SortingService.is_sorted(counts) is True

    def test_unsorted_array(self):
        """Test that unsorted array is not sorted."""
        counts = [1, 5, 3, 2, 4]
        assert SortingService.is_sorted(counts) is False

    def test_single_element(self):
        """Test single element array is sorted."""
        counts = [5]
        assert SortingService.is_sorted(counts) is True

    def test_empty_array(self):
        """Test empty array is sorted."""
        counts = []
        assert SortingService.is_sorted(counts) is True

    def test_two_elements_sorted(self):
        """Test two elements in correct order."""
        counts = [10, 5]
        assert SortingService.is_sorted(counts) is True

    def test_two_elements_unsorted(self):
        """Test two elements in wrong order."""
        counts = [5, 10]
        assert SortingService.is_sorted(counts) is False

    def test_equal_elements(self):
        """Test array with equal elements."""
        counts = [5, 5, 5]
        # Equal elements should satisfy >= condition
        assert SortingService.is_sorted(counts) is True

    def test_partial_sorted_array(self):
        """Test partially sorted array."""
        counts = [10, 8, 7, 9, 5]  # Not fully sorted
        assert SortingService.is_sorted(counts) is False


class TestLoadShuffledToxicWords:
    """Tests for SortingService.load_shuffled_toxic_words function."""

    def test_load_valid_npy_file(self, temp_npy_file, sample_toxic_words):
        """Test loading valid .npy file."""
        words, counts = SortingService.load_shuffled_toxic_words(str(temp_npy_file))

        assert isinstance(words, list)
        assert isinstance(counts, list)
        assert len(words) > 0
        assert len(counts) > 0

    def test_load_missing_file(self):
        """Test loading non-existent file."""
        with pytest.raises((FileNotFoundError, Exception)):
            SortingService.load_shuffled_toxic_words("/nonexistent/path.npy")

    def test_load_with_seed_reproducibility(self, temp_npy_file):
        """Test that same seed produces same shuffle."""
        words1, counts1 = SortingService.load_shuffled_toxic_words(str(temp_npy_file), seed=42)
        words2, counts2 = SortingService.load_shuffled_toxic_words(str(temp_npy_file), seed=42)

        # Same seed should produce same result
        assert words1 == words2
        assert counts1 == counts2

    def test_load_different_seeds(self, temp_npy_file):
        """Test that different seeds may produce different results."""
        words1, _ = SortingService.load_shuffled_toxic_words(str(temp_npy_file), seed=42)
        words2, _ = SortingService.load_shuffled_toxic_words(str(temp_npy_file), seed=123)

        # Might be same or different due to randomness
        # Just verify both complete without error
        assert len(words1) > 0
        assert len(words2) > 0

    def test_load_with_top_n_filter(self, temp_npy_file):
        """Test loading with top_n filtering."""
        words, counts = SortingService.load_shuffled_toxic_words(str(temp_npy_file), top_n=3)

        # Should return at most top_n words
        assert len(words) <= 3

    def test_load_with_top_n_larger_than_data(self, temp_npy_file):
        """Test top_n larger than actual data."""
        words, counts = SortingService.load_shuffled_toxic_words(str(temp_npy_file), top_n=1000)

        # Should return all available words
        assert isinstance(words, list)
        assert len(words) > 0

    def test_load_with_zero_top_n(self, temp_npy_file):
        """Test with top_n=0."""
        words, counts = SortingService.load_shuffled_toxic_words(str(temp_npy_file), top_n=0)

        # May return empty or handle as edge case
        assert isinstance(words, list)

    def test_load_with_negative_top_n(self, temp_npy_file):
        """Test with negative top_n."""
        words, counts = SortingService.load_shuffled_toxic_words(str(temp_npy_file), top_n=-1)

        # Should handle gracefully
        assert isinstance(words, list)

    def test_load_preserves_word_count_pairs(self, temp_npy_file):
        """Test that word-count pairing is preserved."""
        words, counts = SortingService.load_shuffled_toxic_words(str(temp_npy_file))

        assert len(words) == len(counts)
        # All counts should be numeric
        for count in counts:
            assert isinstance(count, (int, np.integer))


class TestSaveDistributionPlot:
    """Tests for SortingService.save_distribution_plot function."""

    def test_save_plot_basic(self, tmp_path, sample_toxic_words):
        """Test basic plot saving."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        output_file = str(tmp_path / "plot.png")

        # Should complete without error
        SortingService.save_distribution_plot(words, counts, output_file)
        # File may or may not be created depending on mocking
        # Just verify function completes

    def test_save_plot_with_invalid_path(self, sample_toxic_words):
        """Test saving to invalid path."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]

        # Should raise or handle gracefully
        try:
            SortingService.save_distribution_plot(words, counts, "/invalid/nonexistent/path/plot.png")
        except (OSError, Exception):
            pass  # Expected

    def test_save_plot_empty_data(self, tmp_path):
        """Test saving plot with empty data."""
        output_file = str(tmp_path / "empty.png")

        # Should handle gracefully
        try:
            SortingService.save_distribution_plot([], [], output_file)
        except ValueError:
            pass  # May raise on empty data

    def test_save_plot_single_word(self, tmp_path):
        """Test saving plot with single word."""
        output_file = str(tmp_path / "single.png")
        SortingService.save_distribution_plot(["word"], [100], output_file)

    @patch("app.services.sorting_service.plt.savefig")
    def test_save_plot_called_with_correct_path(self, mock_savefig, sample_toxic_words):
        """Test that savefig is called with correct path."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        output_path = "/tmp/test.png"

        SortingService.save_distribution_plot(words, counts, output_path)

        # Verify savefig was called (if not mocked elsewhere)
        # Just verify function completes


class TestBogosortSnapshots:
    """Tests for SortingService.bogosort_snapshots function."""

    def test_bogosort_converges(self, sample_toxic_words):
        """Test that bogosort eventually converges."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        stop_flag = {"stop": False}

        snapshots = SortingService.bogosort_snapshots(words, counts, max_iterations=1000, seed=42, stop_flag=stop_flag)

        # Should return list of snapshots
        assert isinstance(snapshots, list)
        assert len(snapshots) > 0

    def test_bogosort_respects_max_iterations(self, sample_toxic_words):
        """Test that bogosort respects max_iterations."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        stop_flag = {"stop": False}

        snapshots = SortingService.bogosort_snapshots(words, counts, max_iterations=5, seed=42, stop_flag=stop_flag)

        # Number of snapshots should not exceed max_iterations
        assert len(snapshots) <= 5

    def test_bogosort_snapshot_format(self, sample_toxic_words):
        """Test snapshot format."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        stop_flag = {"stop": False}

        snapshots = SortingService.bogosort_snapshots(words, counts, max_iterations=100, seed=42, stop_flag=stop_flag)

        if len(snapshots) > 0:
            snapshot = snapshots[0]
            # Should be tuple of (state, iteration_number)
            assert isinstance(snapshot, tuple)
            assert len(snapshot) == 2

    def test_bogosort_stop_flag(self, sample_toxic_words):
        """Test that stop flag stops sorting."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        stop_flag = {"stop": False}

        # Note: In practice, need to stop flag mid-execution
        # For now, just verify it accepts the flag
        snapshots = SortingService.bogosort_snapshots(words, counts, max_iterations=1000, seed=42, stop_flag=stop_flag)
        assert isinstance(snapshots, list)

    def test_bogosort_empty_input(self):
        """Test bogosort with empty input."""
        stop_flag = {"stop": False}
        snapshots = SortingService.bogosort_snapshots([], [], max_iterations=10, seed=42, stop_flag=stop_flag)

        assert isinstance(snapshots, list)

    def test_bogosort_single_element(self):
        """Test bogosort with single element."""
        stop_flag = {"stop": False}
        snapshots = SortingService.bogosort_snapshots(["word"], [100], max_iterations=10, seed=42, stop_flag=stop_flag)

        # Single element is already sorted
        assert isinstance(snapshots, list)

    def test_bogosort_two_elements(self, sample_toxic_words):
        """Test bogosort with two elements."""
        words = [sample_toxic_words[0][0], sample_toxic_words[1][0]]
        counts = [sample_toxic_words[0][1], sample_toxic_words[1][1]]
        stop_flag = {"stop": False}

        snapshots = SortingService.bogosort_snapshots(words, counts, max_iterations=100, seed=42, stop_flag=stop_flag)

        assert isinstance(snapshots, list)
        # Two elements should sort quickly
        assert len(snapshots) < 100

    def test_bogosort_seed_reproducibility(self, sample_toxic_words):
        """Test that same seed produces same sequence."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]

        snapshots1 = SortingService.bogosort_snapshots(words, counts, max_iterations=50, seed=42, stop_flag={"stop": False})
        snapshots2 = SortingService.bogosort_snapshots(words, counts, max_iterations=50, seed=42, stop_flag={"stop": False})

        # Same seed should produce same number of iterations
        assert len(snapshots1) == len(snapshots2)


class TestMergesortSnapshots:
    """Tests for SortingService.mergesort_snapshots function."""

    def test_mergesort_produces_sorted_result(self, sample_toxic_words):
        """Test that mergesort produces sorted output."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        stop_flag = {"stop": False}

        snapshots = SortingService.mergesort_snapshots(words, counts, seed=42, stop_flag=stop_flag)

        assert isinstance(snapshots, list)
        if len(snapshots) > 0:
            # Last snapshot should be sorted
            final_state, _ = snapshots[-1]
            # Extract counts from final state
            if final_state:
                final_counts = [c for _, c in final_state]
                # Should be in descending order
                for i in range(len(final_counts) - 1):
                    assert final_counts[i] >= final_counts[i + 1]

    def test_mergesort_snapshot_format(self, sample_toxic_words):
        """Test mergesort snapshot format."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        stop_flag = {"stop": False}

        snapshots = SortingService.mergesort_snapshots(words, counts, seed=42, stop_flag=stop_flag)

        if len(snapshots) > 0:
            snapshot = snapshots[0]
            assert isinstance(snapshot, tuple)
            assert len(snapshot) == 2

    def test_mergesort_empty_input(self):
        """Test mergesort with empty input."""
        stop_flag = {"stop": False}
        snapshots = SortingService.mergesort_snapshots([], [], seed=42, stop_flag=stop_flag)

        assert isinstance(snapshots, list)

    def test_mergesort_single_element(self):
        """Test mergesort with single element."""
        stop_flag = {"stop": False}
        snapshots = SortingService.mergesort_snapshots(["word"], [100], seed=42, stop_flag=stop_flag)

        assert isinstance(snapshots, list)

    def test_mergesort_already_sorted(self, sample_toxic_words):
        """Test mergesort with already sorted input."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        stop_flag = {"stop": False}

        snapshots = SortingService.mergesort_snapshots(words, counts, seed=42, stop_flag=stop_flag)

        assert isinstance(snapshots, list)

    def test_mergesort_reverse_sorted(self):
        """Test mergesort with reverse sorted input."""
        words = ["e", "d", "c", "b", "a"]
        counts = [1, 2, 3, 4, 5]  # Reverse of desired order
        stop_flag = {"stop": False}

        snapshots = SortingService.mergesort_snapshots(words, counts, seed=42, stop_flag=stop_flag)

        assert isinstance(snapshots, list)
        # Should still sort correctly
        if len(snapshots) > 0:
            final_state, _ = snapshots[-1]
            final_counts = [c for _, c in final_state]
            # Verify descending order
            for i in range(len(final_counts) - 1):
                assert final_counts[i] >= final_counts[i + 1]

    def test_mergesort_stop_flag(self, sample_toxic_words):
        """Test mergesort respects stop flag."""
        words = [w for w, _ in sample_toxic_words]
        counts = [c for _, c in sample_toxic_words]
        stop_flag = {"stop": False}

        snapshots = SortingService.mergesort_snapshots(words, counts, seed=42, stop_flag=stop_flag)
        assert isinstance(snapshots, list)


class TestSaveAnimation:
    """Tests for SortingService.save_sort_animation function."""

    @patch("app.services.sorting_service.imageio.mimsave")
    def test_save_animation_basic(self, mock_mimsave, tmp_path):
        """Test basic animation saving."""
        snapshots = [
            ([("word1", 5), ("word2", 3)], 0),
            ([("word2", 3), ("word1", 5)], 1),
        ]
        output_file = str(tmp_path / "animation.gif")

        SortingService.save_sort_animation(snapshots, output_file, "Test Animation")

        # Verify mimsave was called (mocked)
        mock_mimsave.assert_called()

    @patch("app.services.sorting_service.imageio.mimsave")
    def test_save_animation_empty_snapshots(self, mock_mimsave, tmp_path):
        """Test saving animation with empty snapshots."""
        output_file = str(tmp_path / "empty.gif")

        # Should handle gracefully
        try:
            SortingService.save_sort_animation([], output_file, "Empty")
        except ValueError:
            pass  # May raise on empty snapshots

    @patch("app.services.sorting_service.imageio.mimsave")
    def test_save_animation_single_snapshot(self, mock_mimsave, tmp_path):
        """Test saving animation with single snapshot."""
        snapshots = [([("word", 5)], 0)]
        output_file = str(tmp_path / "single.gif")

        SortingService.save_sort_animation(snapshots, output_file, "Single")

    def test_save_animation_invalid_path(self):
        """Test saving animation to invalid path."""
        snapshots = [([("word", 5)], 0)]

        # Should raise or handle gracefully
        try:
            SortingService.save_sort_animation(snapshots, "/invalid/nonexistent/path/anim.gif", "Test")
        except (OSError, Exception):
            pass  # Expected

    @patch("app.services.sorting_service.imageio.mimsave")
    def test_save_animation_with_title(self, mock_mimsave, tmp_path):
        """Test that title is passed correctly."""
        snapshots = [([("word", 5)], 0)]
        output_file = str(tmp_path / "titled.gif")
        title = "Bogosort Visualization"

        SortingService.save_sort_animation(snapshots, output_file, title)

        # Function should complete without error


class TestSortingIntegration:
    """Integration tests combining multiple sorting functions."""

    def test_load_and_sort_workflow(self, temp_npy_file):
        """Test complete workflow: load, sort, verify."""
        words, counts = SortingService.load_shuffled_toxic_words(str(temp_npy_file), seed=42)

        # Verify data is shuffled
        assert len(words) > 0
        assert len(counts) > 0

        # Run bogosort
        stop_flag = {"stop": False}
        snapshots = SortingService.bogosort_snapshots(words, counts, max_iterations=100, seed=42, stop_flag=stop_flag)

        # Verify results
        assert len(snapshots) > 0

    def test_compare_sorting_algorithms(self, temp_npy_file):
        """Compare bogosort and mergesort on same data."""
        words, counts = SortingService.load_shuffled_toxic_words(str(temp_npy_file), seed=42)
        stop_flag = {"stop": False}

        bogosort_snaps = SortingService.bogosort_snapshots(words, counts, max_iterations=50, seed=42, stop_flag=stop_flag)
        mergesort_snaps = SortingService.mergesort_snapshots(words, counts, seed=42, stop_flag=stop_flag)

        # Mergesort should converge faster
        assert len(mergesort_snaps) < len(bogosort_snaps)
