"""
Improved comprehensive test suite for BubbleVisualizer with 4K canvas and placeholder.txt dataset.
All tests use 4K canvas and the placeholder.txt dataset with proper word counting algorithm.
"""

import pytest
import tempfile
import os
import sys
import string
from collections import Counter
from typing import List, Optional
from unittest.mock import patch, Mock
import subprocess

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bubble_visualizer import BubbleVisualizer
from word_analysis import WordAnalyzer
from dictionary_manager import DictionaryManager
from text_extractor import TextExtractorFactory
from config import ANALYSIS_CONFIG

# Import TestDataLoader from tests directory
tests_dir = os.path.dirname(__file__)
sys.path.append(tests_dir)
from test_data_loader import TestDataLoader


@pytest.fixture
def placeholder_dataset():
    """Fixture providing the placeholder.txt dataset."""
    return TestDataLoader.load_placeholder_dataset()


@pytest.fixture
def real_dict_manager():
    """Fixture providing a real dictionary manager."""
    return TestDataLoader.get_real_dict_manager()


@pytest.fixture
def bubble_visualizer_4k():
    """Fixture providing a 4K BubbleVisualizer instance."""
    return BubbleVisualizer()  # Uses default 4K dimensions


@pytest.fixture
def temp_output_file():
    """Fixture that provides a temporary output file path and cleans it up after test."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        output_path = tmp_file.name
    yield output_path
    # Cleanup
    if os.path.exists(output_path):
        os.unlink(output_path)


@pytest.mark.unit
class TestBubbleVisualizerInitialization:
    """Test BubbleVisualizer initialization and basic properties."""

    def test_default_4k_initialization(self):
        """Test that default initialization creates 4K bubble visualizer."""
        visualizer = BubbleVisualizer()
        assert visualizer.width == 3840, "Default width should be 4K (3840)"
        assert visualizer.height == 2160, "Default height should be 4K (2160)"
        assert (
            visualizer.bubbles == []
        ), "Bubbles list should be empty on initialization"

    def test_custom_dimensions_initialization(self):
        """Test initialization with custom dimensions."""
        width, height = 1920, 1080
        visualizer = BubbleVisualizer(width=width, height=height)
        assert visualizer.width == width
        assert visualizer.height == height
        assert visualizer.bubbles == []

    def test_word_type_colors_defined(self):
        """Test that word type colors are properly defined for distinct coloring."""
        visualizer = BubbleVisualizer()
        assert hasattr(visualizer, "WORD_TYPE_COLORS")
        assert isinstance(visualizer.WORD_TYPE_COLORS, dict)
        assert len(visualizer.WORD_TYPE_COLORS) >= 15  # Should have many word types

        # Check that all colors are valid hex codes
        for word_type, color in visualizer.WORD_TYPE_COLORS.items():
            assert color.startswith("#"), f"Color for {word_type} should start with #"
            assert len(color) == 7, f"Color for {word_type} should be #RRGGBB format"

    @patch("bubble_visualizer.PIL_AVAILABLE", False)
    def test_initialization_without_pil_raises_error(self):
        """Test that initialization without PIL raises ImportError."""
        with pytest.raises(ImportError, match="PIL \\(Pillow\\) is required"):
            BubbleVisualizer()

    def test_get_available_word_types_static_method(self):
        """Test that get_available_word_types returns expected word types."""
        word_types = BubbleVisualizer.get_available_word_types()
        assert isinstance(word_types, list)
        assert len(word_types) > 10  # Should have many word types
        assert "noun" in word_types
        assert "verb" in word_types
        assert "adj" in word_types


@pytest.mark.quality
class TestBubbleVisualizerSpaceUtilization:
    """Test space utilization requirements with 4K canvas and placeholder.txt dataset."""

    def test_reasonable_space_utilization_at_least_50_percent(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that bubble visualizer achieves at least 50% space utilization with 4K canvas."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, *args, **kwargs):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]
            # Mock the actual image creation
            pass

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            # Patch the _create_image method to capture bubbles
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_positioned_bubbles

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        # Calculate space utilization
        canvas_area = bubble_visualizer_4k.width * bubble_visualizer_4k.height
        used_area = self._calculate_used_area(positioned_bubbles)
        utilization = used_area / canvas_area

        print(f"\nSpace Utilization Test Results:")
        print(
            f"Canvas: {bubble_visualizer_4k.width}x{bubble_visualizer_4k.height} (4K)"
        )
        print(f"Total words in dataset: {len(placeholder_dataset)}")
        print(f"Bubbles placed: {len(positioned_bubbles)}")
        print(f"Space utilization: {utilization:.2%}")
        print(f"Used area: {used_area:,.0f} pixels")
        print(f"Canvas area: {canvas_area:,.0f} pixels")

        # Requirement: At least 50% space utilization
        assert (
            utilization >= 0.50
        ), f"Space utilization must be at least 50%, got {utilization:.2%}"

    def test_reasonable_space_utilization_with_word_types_excluded(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test space utilization with word types excluded is still at least 50%."""
        positioned_bubbles = []
        exclude_types = ["art", "conj", "prep"]  # Common words to exclude

        def capture_positioned_bubbles(bubbles, *args, **kwargs):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_positioned_bubbles

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                exclude_types=exclude_types,
            )

        # Calculate space utilization
        canvas_area = bubble_visualizer_4k.width * bubble_visualizer_4k.height
        used_area = self._calculate_used_area(positioned_bubbles)
        utilization = used_area / canvas_area

        print(f"\nSpace Utilization Test Results (Excluded Types: {exclude_types}):")
        print(
            f"Canvas: {bubble_visualizer_4k.width}x{bubble_visualizer_4k.height} (4K)"
        )
        print(f"Total words in dataset: {len(placeholder_dataset)}")
        print(f"Bubbles placed: {len(positioned_bubbles)}")
        print(f"Space utilization: {utilization:.2%}")

        # Requirement: At least 50% space utilization even with exclusions
        assert (
            utilization >= 0.50
        ), f"Space utilization with excluded types must be at least 50%, got {utilization:.2%}"

    def _calculate_used_area(self, positioned_bubbles):
        """Calculate total area used by bubbles."""
        total_area = 0
        for bubble in positioned_bubbles:
            if len(bubble) >= 7:  # (word, count, word_type, color, radius, x, y)
                radius = bubble[4]
                total_area += 3.14159 * radius * radius
        return total_area


@pytest.mark.quality
class TestBubbleVisualizerBubbleCount:
    """Test bubble count requirements with 4K canvas and placeholder.txt dataset."""

    def test_reasonable_amount_of_bubbles_at_least_500(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that at least 500 bubbles are placed from the base dataset."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, *args, **kwargs):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_positioned_bubbles

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        print(f"\nBubble Count Test Results:")
        print(f"Total words in dataset: {len(placeholder_dataset)}")
        print(f"Bubbles placed: {len(positioned_bubbles)}")
        print(f"Placement rate: {len(positioned_bubbles)/len(placeholder_dataset):.2%}")

        # Requirement: At least 500 bubbles placed
        assert (
            len(positioned_bubbles) >= 500
        ), f"Must place at least 500 bubbles from base dataset, got {len(positioned_bubbles)}"

    def test_reasonable_amount_of_bubbles_with_word_types_excluded(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that at least 500 bubbles are placed even with word types excluded."""
        positioned_bubbles = []
        exclude_types = ["art", "conj", "prep"]

        def capture_positioned_bubbles(bubbles, *args, **kwargs):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_positioned_bubbles

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                exclude_types=exclude_types,
            )

        print(f"\nBubble Count Test Results (Excluded Types: {exclude_types}):")
        print(f"Total words in dataset: {len(placeholder_dataset)}")
        print(f"Bubbles placed: {len(positioned_bubbles)}")

        # Requirement: At least 500 bubbles even with exclusions
        assert (
            len(positioned_bubbles) >= 500
        ), f"Must place at least 500 bubbles even with excluded types, got {len(positioned_bubbles)}"


@pytest.mark.integration
class TestBubbleVisualizerImageProperties:
    """Test image properties and quality requirements."""

    def test_produced_image_is_4k(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that produced image is 4K resolution."""
        mock_img = Mock()

        with patch("bubble_visualizer.Image") as mock_image, patch(
            "bubble_visualizer.ImageDraw"
        ):
            mock_image.new.return_value = mock_img

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

            # Verify image was created with 4K dimensions
            mock_image.new.assert_called_once()
            call_args = mock_image.new.call_args

            # Check that 4K dimensions were used
            assert call_args[0][1] == (
                3840,
                2160,
            ), f"Image must be created with 4K dimensions (3840x2160), got {call_args[0][1]}"

            print(f"\n4K Image Test Results:")
            print(f"Image dimensions: {call_args[0][1]} (4K confirmed)")

    def test_produced_image_uses_distinct_colors_for_different_word_types(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that different word types use distinct colors."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, *args, **kwargs):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_positioned_bubbles

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        # Collect word types and their colors
        word_type_colors = {}
        for bubble in positioned_bubbles:
            if len(bubble) >= 4:  # (word, count, word_type, color, ...)
                word_type = bubble[2]
                color = bubble[3]
                if word_type in word_type_colors:
                    # Verify consistency: same word type should always have same color
                    assert word_type_colors[word_type] == color, (
                        f"Word type '{word_type}' has inconsistent colors: "
                        f"{word_type_colors[word_type]} vs {color}"
                    )
                else:
                    word_type_colors[word_type] = color

        print(f"\nColor Distinctness Test Results:")
        print(f"Unique word types found: {len(word_type_colors)}")
        print(f"Word type to color mapping:")
        for word_type, color in sorted(word_type_colors.items()):
            print(f"  {word_type}: {color}")

        # Verify we have multiple word types with distinct colors
        assert (
            len(word_type_colors) >= 5
        ), f"Must have at least 5 distinct word types with colors, got {len(word_type_colors)}"

        # Verify all colors are distinct
        colors = list(word_type_colors.values())
        unique_colors = set(colors)
        assert len(colors) == len(
            unique_colors
        ), f"All word types must have distinct colors, found duplicates"


@pytest.mark.quality
class TestBubbleVisualizerQualityAssurance:
    """Test quality assurance requirements."""

    def test_bubbles_do_not_overlap_with_base_dataset_at_4k(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that bubbles do not overlap with the base dataset at 4K."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, *args, **kwargs):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_positioned_bubbles

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        # Check for overlaps
        overlap_count = self._count_overlaps(positioned_bubbles)

        print(f"\nOverlap Test Results:")
        print(f"Total bubbles: {len(positioned_bubbles)}")
        print(f"Overlapping pairs: {overlap_count}")
        print(
            f"Overlap rate: {overlap_count/(len(positioned_bubbles)*(len(positioned_bubbles)-1)/2)*100:.2f}%"
            if len(positioned_bubbles) > 1
            else "N/A"
        )

        # Requirement: No bubbles should overlap
        assert (
            overlap_count == 0
        ), f"Bubbles must not overlap, found {overlap_count} overlapping pairs"

    def test_bubbles_have_varying_size_based_on_frequency(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that bubbles have varying sizes based on word frequency."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, *args, **kwargs):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_positioned_bubbles

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        # Extract frequencies and radii
        frequency_radius_pairs = []
        for bubble in positioned_bubbles:
            if len(bubble) >= 5:  # (word, count, word_type, color, radius, ...)
                frequency = bubble[1]
                radius = bubble[4]
                frequency_radius_pairs.append((frequency, radius))

        # Sort by frequency
        frequency_radius_pairs.sort(key=lambda x: x[0], reverse=True)

        print(f"\nSize Variation Test Results:")
        print(f"Bubble count: {len(frequency_radius_pairs)}")
        print(
            f"Frequency range: {frequency_radius_pairs[-1][0]} - {frequency_radius_pairs[0][0]}"
        )
        print(
            f"Radius range: {min(r for f, r in frequency_radius_pairs)} - {max(r for f, r in frequency_radius_pairs)}"
        )

        # Check that radii generally increase with frequency
        # Allow some tolerance for placement algorithm optimizations
        correct_size_ordering = 0
        total_comparisons = 0

        for i in range(len(frequency_radius_pairs) - 1):
            freq1, radius1 = frequency_radius_pairs[i]
            freq2, radius2 = frequency_radius_pairs[i + 1]

            if freq1 > freq2:  # Higher frequency should have larger or equal radius
                total_comparisons += 1
                if radius1 >= radius2:
                    correct_size_ordering += 1

        if total_comparisons > 0:
            ordering_accuracy = correct_size_ordering / total_comparisons
            print(f"Size ordering accuracy: {ordering_accuracy:.2%}")

            # Requirement: At least 80% of comparisons should show correct size ordering
            assert (
                ordering_accuracy >= 0.80
            ), f"Bubble sizes must generally reflect frequency (80% accuracy required), got {ordering_accuracy:.2%}"

    def test_does_not_skip_words_before_cutoff(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that important words are not skipped before the frequency cutoff."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, *args, **kwargs):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_positioned_bubbles

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        # Get the words that were placed and their frequencies
        placed_words = {
            bubble[0]: bubble[1] for bubble in positioned_bubbles if len(bubble) >= 2
        }

        # Get the most frequent words from dataset
        most_frequent_words = placeholder_dataset.most_common(len(placed_words))

        # Check that high-frequency words are not missing
        missing_high_freq_words = []
        for word, freq in most_frequent_words[:100]:  # Check top 100 words
            if word not in placed_words:
                missing_high_freq_words.append((word, freq))

        print(f"\nWord Cutoff Test Results:")
        print(f"Dataset size: {len(placeholder_dataset)} unique words")
        print(f"Words placed: {len(placed_words)}")
        print(f"Missing high-frequency words (top 100): {len(missing_high_freq_words)}")

        if missing_high_freq_words:
            print("Missing words:")
            for word, freq in missing_high_freq_words[:10]:  # Show first 10
                print(f"  {word}: {freq}")

        # Requirement: Should not skip many high-frequency words
        # Allow some flexibility for placement algorithm limitations
        missing_rate = len(missing_high_freq_words) / 100
        assert (
            missing_rate <= 0.20
        ), f"Should not skip more than 20% of top 100 words, missing {missing_rate:.2%}"

    def _count_overlaps(self, positioned_bubbles):
        """Count the number of overlapping bubble pairs."""
        overlap_count = 0
        for i in range(len(positioned_bubbles)):
            for j in range(i + 1, len(positioned_bubbles)):
                if self._bubbles_overlap(positioned_bubbles[i], positioned_bubbles[j]):
                    overlap_count += 1
        return overlap_count

    def _bubbles_overlap(self, bubble1, bubble2):
        """Check if two bubbles overlap."""
        if len(bubble1) < 7 or len(bubble2) < 7:
            return False

        # Extract positions and radii (word, count, word_type, color, radius, x, y)
        x1, y1, r1 = bubble1[5], bubble1[6], bubble1[4]
        x2, y2, r2 = bubble2[5], bubble2[6], bubble2[4]

        # Calculate distance between centers
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        # Bubbles overlap if distance < sum of radii
        return distance < (r1 + r2)


@pytest.mark.unit
class TestBubbleVisualizerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_word_counts(
        self, bubble_visualizer_4k, real_dict_manager, temp_output_file
    ):
        """Test handling of empty word counts."""
        empty_counts = Counter()

        # Should handle empty input gracefully without crashing
        bubble_visualizer_4k.create_bubble_chart(
            word_counts=empty_counts,
            dict_manager=real_dict_manager,
            output_path=temp_output_file,
        )
        # Test passes if no exception is raised

    def test_single_word_count(
        self, bubble_visualizer_4k, real_dict_manager, temp_output_file
    ):
        """Test handling of single word count."""
        single_word = Counter({"test": 5})

        # Should handle single word gracefully
        bubble_visualizer_4k.create_bubble_chart(
            word_counts=single_word,
            dict_manager=real_dict_manager,
            output_path=temp_output_file,
        )
        # Test passes if no exception is raised

    def test_exclude_all_types(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test behavior when excluding all word types."""
        all_types = BubbleVisualizer.get_available_word_types()

        # Should handle exclusion of all types gracefully
        bubble_visualizer_4k.create_bubble_chart(
            word_counts=placeholder_dataset,
            dict_manager=real_dict_manager,
            output_path=temp_output_file,
            exclude_types=all_types,
        )
        # Test passes if no exception is raised


@pytest.mark.cli
class TestBubbleVisualizerCLI:
    """Test CLI functionality."""

    def test_cli_arguments_work(self, temp_output_file):
        """Test that CLI arguments work correctly."""
        # Test basic bubble chart creation via CLI
        cmd = [
            sys.executable,
            "main.py",
            "--input",
            "texts/placeholder.txt",
            "--build-graph",
            temp_output_file,
            "--quiet",
        ]

        # Set environment to handle unicode characters properly
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace problematic characters
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
        )

        print(f"\nCLI Test Results:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")
        print(f"Stdout length: {len(result.stdout)} characters")
        if result.stderr:
            print(f"Stderr: {result.stderr}")

        # Requirement: CLI should work without errors
        assert (
            result.returncode == 0
        ), f"CLI command failed with return code {result.returncode}"

        # Check that output file was created
        assert os.path.exists(temp_output_file), "CLI should create output file"

        # Check file size
        file_size = os.path.getsize(temp_output_file)
        assert (
            file_size > 100000
        ), f"Output file seems too small: {file_size} bytes (expected >100KB for 4K image)"

        print(f"SUCCESS: CLI working correctly, output file: {file_size:,} bytes")

    def test_cli_with_exclude_types(self, temp_output_file):
        """Test CLI with exclude types functionality."""
        cmd = [
            sys.executable,
            "main.py",
            "--input",
            "texts/placeholder.txt",
            "--build-graph",
            temp_output_file,
            "--exclude-types",
            "art",
            "conj",
            "prep",
            "--quiet",
        ]

        # Set environment to handle unicode characters properly
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace problematic characters
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
        )

        print(f"\nCLI Exclude Types Test Results:")
        print(f"Return code: {result.returncode}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")

        # Requirement: CLI should work with exclude types
        assert (
            result.returncode == 0
        ), f"CLI with exclude types failed with return code {result.returncode}"
        assert os.path.exists(
            temp_output_file
        ), "CLI with exclude types should create output file"

        # Check file size
        file_size = os.path.getsize(temp_output_file)
        print(
            f"SUCCESS: CLI with exclusions working correctly, output file: {file_size:,} bytes"
        )

    def test_cli_list_types(self):
        """Test CLI --list-types functionality."""
        cmd = [sys.executable, "main.py", "--list-types"]

        # Set environment to handle unicode characters properly
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Replace problematic characters
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
        )

        print(f"\nCLI List Types Test Results:")
        print(f"Return code: {result.returncode}")

        assert result.returncode == 0, "CLI --list-types should work"
        assert (
            "word types" in result.stdout.lower()
        ), "Should show word types information"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
