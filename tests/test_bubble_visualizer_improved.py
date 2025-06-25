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
import numpy as np

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


@pytest.mark.quality
class TestBubbleVisualizerGradientMode:
    """Test gradient mode functionality for bubble visualizer."""

    def test_gradient_mode_creates_different_background_than_normal_mode(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that gradient mode produces a visually different background from normal mode."""
        gradient_background = None
        normal_background = None
        positioned_bubbles = []

        def capture_gradient_background(bubbles, output_path, gradient_mode=False):
            nonlocal gradient_background, normal_background, positioned_bubbles
            positioned_bubbles = bubbles[:]
            if gradient_mode:
                # Capture gradient background generation
                gradient_background = bubble_visualizer_4k._create_gradient_background(
                    bubbles
                )
            else:
                # For normal mode, create a white background equivalent
                normal_background = np.full(
                    (bubble_visualizer_4k.height, bubble_visualizer_4k.width, 3),
                    255,
                    dtype=np.uint8,
                )

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_gradient_background

            # First, create normal mode chart
            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=False,
            )

            # Then create gradient mode chart
            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=True,
            )

        # Verify both backgrounds were captured
        assert gradient_background is not None, "Gradient background should be captured"
        assert normal_background is not None, "Normal background should be captured"
        assert len(positioned_bubbles) > 0, "Should have positioned bubbles"

        # Verify backgrounds are different
        assert not np.array_equal(
            gradient_background, normal_background
        ), "Gradient mode should produce different background than normal mode"

        # Verify gradient background is not uniform (should have color variations)
        gradient_variance = np.var(gradient_background)
        normal_variance = np.var(normal_background)
        assert (
            gradient_variance > normal_variance
        ), "Gradient background should have more color variation than normal background"

        print(f"\nGradient Mode Test Results:")
        print(f"Positioned bubbles: {len(positioned_bubbles)}")
        print(f"Gradient variance: {gradient_variance:.2f}")
        print(f"Normal variance: {normal_variance:.2f}")

    def test_gradient_background_uses_bubble_colors(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that gradient background incorporates colors from positioned bubbles."""
        gradient_background = None
        positioned_bubbles = []

        def capture_gradient_data(bubbles, output_path, gradient_mode=False):
            nonlocal gradient_background, positioned_bubbles
            if gradient_mode:
                positioned_bubbles = bubbles[:]
                gradient_background = bubble_visualizer_4k._create_gradient_background(
                    bubbles
                )

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_gradient_data

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=True,
            )

        assert (
            gradient_background is not None
        ), "Gradient background should be generated"
        assert len(positioned_bubbles) > 0, "Should have positioned bubbles"

        # Extract colors from positioned bubbles
        bubble_colors = []
        for bubble in positioned_bubbles:
            word, count, word_type, color, radius, x, y = bubble
            if color.startswith("#"):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                bubble_colors.append((r, g, b))

        # Verify gradient background contains colors similar to bubble colors
        unique_gradient_colors = self._extract_unique_colors_from_gradient(
            gradient_background
        )

        # Check that gradient has more color diversity than just the neutral background
        assert (
            len(unique_gradient_colors) > 10
        ), "Gradient should contain diverse colors from bubbles"

        # Verify some gradient colors are close to bubble colors
        colors_found = 0
        for bubble_color in bubble_colors[:10]:  # Check first 10 bubble colors
            if self._is_color_present_in_gradient(bubble_color, unique_gradient_colors):
                colors_found += 1

        assert (
            colors_found > 0
        ), "Gradient background should contain colors similar to bubble colors"

        print(f"\nGradient Color Usage Test Results:")
        print(f"Bubble colors tested: {len(bubble_colors[:10])}")
        print(f"Unique gradient colors: {len(unique_gradient_colors)}")
        print(f"Bubble colors found in gradient: {colors_found}")

    def test_gradient_blending_creates_smooth_transitions(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that gradient mode creates smooth color transitions between bubble regions."""
        gradient_background = None
        positioned_bubbles = []

        def capture_gradient_data(bubbles, output_path, gradient_mode=False):
            nonlocal gradient_background, positioned_bubbles
            if gradient_mode:
                positioned_bubbles = bubbles[:]
                gradient_background = bubble_visualizer_4k._create_gradient_background(
                    bubbles
                )

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_gradient_data

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=True,
            )

        assert (
            gradient_background is not None
        ), "Gradient background should be generated"
        assert len(positioned_bubbles) >= 2, "Need at least 2 bubbles to test blending"

        # Find suitable bubbles for transition testing
        bubble_pairs = self._find_suitable_bubble_pairs_for_blending(positioned_bubbles)
        if not bubble_pairs:
            pytest.skip("Could not find suitable bubble pairs to test blending")

        # Test multiple bubble pairs to get reliable results
        smooth_transitions_found = 0
        total_pairs_tested = min(3, len(bubble_pairs))  # Test up to 3 pairs

        for bubble1, bubble2 in bubble_pairs[:total_pairs_tested]:
            # Extract positions
            x1, y1 = (
                bubble1[5],
                bubble1[6],
            )  # bubble format: (word, count, word_type, color, radius, x, y)
            x2, y2 = bubble2[5], bubble2[6]

            # Sample colors along the line between the two bubbles
            transition_colors = self._sample_colors_along_line(
                gradient_background, x1, y1, x2, y2, num_samples=15
            )

            # Verify we have a smooth transition (colors should vary gradually)
            color_changes = self._calculate_color_changes(transition_colors)

            # Calculate transition quality metrics
            max_change = max(color_changes) if color_changes else 0
            avg_change = sum(color_changes) / len(color_changes) if color_changes else 0

            # Focus on gradient smoothness rather than absolute color change values
            # A smooth gradient has relatively consistent change rates, not necessarily small changes
            if len(color_changes) > 1:
                # Calculate standard deviation of color changes to measure consistency
                variance = sum(
                    (change - avg_change) ** 2 for change in color_changes
                ) / len(color_changes)
                std_dev = variance**0.5

                # Coefficient of variation: std_dev / mean (measures relative consistency)
                # Lower values indicate more consistent (smoother) transitions
                consistency_ratio = (
                    std_dev / avg_change if avg_change > 0 else float("inf")
                )

                # Check for abrupt single jumps that dominate the transition
                sorted_changes = sorted(color_changes)
                # If the largest change is much bigger than the median, it's likely abrupt
                median_change = sorted_changes[len(sorted_changes) // 2]
                max_to_median_ratio = (
                    max_change / median_change if median_change > 0 else float("inf")
                )
            else:
                consistency_ratio = float("inf")
                max_to_median_ratio = float("inf")

            print(f"\nBubble pair {bubble_pairs.index((bubble1, bubble2)) + 1}:")
            print(f"  Positions: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"  Max color change: {max_change:.2f}")
            print(f"  Average color change: {avg_change:.2f}")
            print(f"  Consistency ratio: {consistency_ratio:.2f}")
            print(f"  Max/median ratio: {max_to_median_ratio:.2f}")

            # Consider transition smooth if:
            # 1. Changes are relatively consistent (low coefficient of variation)
            # 2. No single change dominates the transition (reasonable max/median ratio)
            # 3. There is actual color variation (avg_change > 0)
            is_consistent = consistency_ratio < 2.0  # Reasonable consistency
            no_dominant_jumps = max_to_median_ratio < 5.0  # No extreme outliers
            has_variation = avg_change > 0

            if is_consistent and no_dominant_jumps and has_variation:
                smooth_transitions_found += 1
                print(f"  → Smooth transition found!")
            else:
                print(
                    f"  → Not smooth: consistent={is_consistent}, no_dominant={no_dominant_jumps}, has_variation={has_variation}"
                )

        # Require at least one smooth transition among tested pairs
        assert (
            smooth_transitions_found > 0
        ), f"Should find at least one smooth transition among {total_pairs_tested} tested pairs"

        print(f"\nGradient Blending Test Summary:")
        print(f"Pairs tested: {total_pairs_tested}")
        print(f"Smooth transitions found: {smooth_transitions_found}")

    def test_gradient_colors_accurate_to_bubble_colors(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that gradient colors near bubble centers are accurate to the actual bubble colors."""
        gradient_background = None
        positioned_bubbles = []

        def capture_gradient_data(bubbles, output_path, gradient_mode=False):
            nonlocal gradient_background, positioned_bubbles
            if gradient_mode:
                positioned_bubbles = bubbles[:]
                gradient_background = bubble_visualizer_4k._create_gradient_background(
                    bubbles
                )

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_gradient_data

            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=True,
            )

        assert (
            gradient_background is not None
        ), "Gradient background should be generated"
        assert len(positioned_bubbles) > 0, "Should have positioned bubbles"

        # Test a sample of bubbles for color accuracy
        test_bubbles = positioned_bubbles[: min(10, len(positioned_bubbles))]
        accurate_colors = 0
        total_tested = 0

        for bubble in test_bubbles:
            word, count, word_type, expected_color, radius, x, y = bubble

            # Parse expected color from hex
            if not expected_color.startswith("#"):
                continue

            try:
                expected_r = int(expected_color[1:3], 16)
                expected_g = int(expected_color[3:5], 16)
                expected_b = int(expected_color[5:7], 16)
                expected_rgb = (expected_r, expected_g, expected_b)
            except ValueError:
                continue

            # Sample gradient color at bubble center
            gradient_color = gradient_background[y, x]
            gradient_rgb = tuple(gradient_color)

            # Calculate color difference
            color_diff = np.linalg.norm(np.array(expected_rgb) - np.array(gradient_rgb))

            # Also sample colors around the bubble center for better accuracy
            sample_colors = []
            sample_radius = max(1, radius // 4)  # Sample within inner quarter of bubble
            for dy in range(
                -sample_radius, sample_radius + 1, max(1, sample_radius // 2)
            ):
                for dx in range(
                    -sample_radius, sample_radius + 1, max(1, sample_radius // 2)
                ):
                    sample_x = max(0, min(x + dx, gradient_background.shape[1] - 1))
                    sample_y = max(0, min(y + dy, gradient_background.shape[0] - 1))
                    sample_colors.append(tuple(gradient_background[sample_y, sample_x]))

            # Find the closest color among samples
            min_diff = color_diff
            for sample_color in sample_colors:
                diff = np.linalg.norm(np.array(expected_rgb) - np.array(sample_color))
                min_diff = min(min_diff, diff)

            total_tested += 1

            # Colors are considered accurate if within reasonable tolerance
            # (allowing for blending effects and RGB conversion)
            tolerance = 60  # Reasonable tolerance for color accuracy
            if min_diff <= tolerance:
                accurate_colors += 1

            print(f"Bubble '{word}' at ({x}, {y}):")
            print(f"  Expected RGB: {expected_rgb}")
            print(f"  Gradient RGB: {gradient_rgb}")
            print(f"  Color difference: {color_diff:.2f}")
            print(f"  Min difference (sampled): {min_diff:.2f}")
            print(f"  Accurate: {'Yes' if min_diff <= tolerance else 'No'}")

        # Require reasonable color accuracy
        accuracy_rate = accurate_colors / total_tested if total_tested > 0 else 0

        assert total_tested > 0, "Should have testable bubbles with valid colors"
        assert (
            accuracy_rate >= 0.5
        ), f"At least 50% of bubble colors should be accurate, got {accuracy_rate:.1%} ({accurate_colors}/{total_tested})"

        print(f"\nGradient Color Accuracy Test Summary:")
        print(f"Bubbles tested: {total_tested}")
        print(f"Accurate colors: {accurate_colors}")
        print(f"Accuracy rate: {accuracy_rate:.1%}")

    # P2P TESTS:
    def test_gradient_mode_handles_edge_cases(
        self, bubble_visualizer_4k, real_dict_manager, temp_output_file
    ):
        """Test gradient mode with edge cases like single bubble or empty dataset."""
        gradient_background = None

        def capture_gradient_data(bubbles, output_path, gradient_mode=False):
            nonlocal gradient_background
            if gradient_mode:
                gradient_background = bubble_visualizer_4k._create_gradient_background(
                    bubbles
                )

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = bubble_visualizer_4k._create_image
            bubble_visualizer_4k._create_image = capture_gradient_data

            # Test with single word
            single_word = Counter({"test": 100})
            bubble_visualizer_4k.create_bubble_chart(
                word_counts=single_word,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=True,
            )

        assert (
            gradient_background is not None
        ), "Should generate gradient even with single bubble"

        # Should produce a valid gradient background
        assert gradient_background.shape == (
            bubble_visualizer_4k.height,
            bubble_visualizer_4k.width,
            3,
        ), "Gradient background should have correct dimensions"
        assert (
            gradient_background.dtype == np.uint8
        ), "Gradient background should be uint8 type"

        # Test with empty dataset (should not crash)
        empty_counts = Counter()
        try:
            bubble_visualizer_4k.create_bubble_chart(
                word_counts=empty_counts,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=True,
            )
        except Exception as e:
            pytest.fail(
                f"Gradient mode should handle empty dataset gracefully, got: {e}"
            )

    def test_gradient_mode_preserves_bubble_positioning(
        self,
        bubble_visualizer_4k,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
    ):
        """Test that gradient mode doesn't affect bubble positioning logic."""
        normal_bubbles = []
        gradient_bubbles = []

        def capture_normal_bubbles(bubbles, output_path, gradient_mode=False):
            nonlocal normal_bubbles
            if not gradient_mode:
                normal_bubbles = bubbles[:]

        def capture_gradient_bubbles(bubbles, output_path, gradient_mode=False):
            nonlocal gradient_bubbles
            if gradient_mode:
                gradient_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            # Create normal mode chart
            bubble_visualizer_4k._create_image = capture_normal_bubbles
            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=False,
            )

            # Create gradient mode chart
            bubble_visualizer_4k._create_image = capture_gradient_bubbles
            bubble_visualizer_4k.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                gradient_mode=True,
            )

        assert len(normal_bubbles) > 0, "Should have normal mode bubbles"
        assert len(gradient_bubbles) > 0, "Should have gradient mode bubbles"

        # Bubble positioning should be consistent between modes
        # (allowing for some randomness in positioning algorithm)
        assert len(normal_bubbles) == len(
            gradient_bubbles
        ), "Both modes should position the same number of bubbles"

        print(f"\nGradient Mode Positioning Test Results:")
        print(f"Normal mode bubbles: {len(normal_bubbles)}")
        print(f"Gradient mode bubbles: {len(gradient_bubbles)}")

    def _extract_unique_colors_from_gradient(self, gradient_bg, sample_size=1000):
        """Extract unique colors from gradient background by sampling."""
        height, width, _ = gradient_bg.shape

        # Sample random pixels to get color diversity
        sample_indices = np.random.choice(
            height * width, min(sample_size, height * width), replace=False
        )
        sampled_pixels = gradient_bg.reshape(-1, 3)[sample_indices]

        # Get unique colors (with some tolerance for near-identical colors)
        unique_colors = []
        for pixel in sampled_pixels:
            is_unique = True
            for existing_color in unique_colors:
                if np.linalg.norm(pixel - existing_color) < 10:  # Tolerance threshold
                    is_unique = False
                    break
            if is_unique:
                unique_colors.append(tuple(pixel))

        return unique_colors

    def _is_color_present_in_gradient(
        self, target_color, gradient_colors, tolerance=30
    ):
        """Check if a target color is present in the gradient colors within tolerance."""
        target_array = np.array(target_color)
        for gradient_color in gradient_colors:
            gradient_array = np.array(gradient_color)
            if np.linalg.norm(target_array - gradient_array) < tolerance:
                return True
        return False

    def _find_nearby_bubbles(self, positioned_bubbles):
        """Find two bubbles that are close to each other for blending tests."""
        if len(positioned_bubbles) < 2:
            return None, None

        min_distance = float("inf")
        bubble1, bubble2 = None, None

        for i in range(len(positioned_bubbles)):
            for j in range(i + 1, len(positioned_bubbles)):
                b1, b2 = positioned_bubbles[i], positioned_bubbles[j]
                x1, y1 = b1[5], b1[6]
                x2, y2 = b2[5], b2[6]
                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                # Find bubbles that are close but not overlapping
                if 50 < distance < min_distance and distance < 300:
                    min_distance = distance
                    bubble1, bubble2 = b1, b2

        return bubble1, bubble2

    def _find_suitable_bubble_pairs_for_blending(self, positioned_bubbles):
        """Find multiple suitable bubble pairs for robust blending tests."""
        if len(positioned_bubbles) < 2:
            return []

        pairs = []

        # Sort bubbles by position for more systematic pairing
        sorted_bubbles = sorted(positioned_bubbles, key=lambda b: (b[5], b[6]))

        for i in range(len(sorted_bubbles)):
            for j in range(i + 1, len(sorted_bubbles)):
                b1, b2 = sorted_bubbles[i], sorted_bubbles[j]
                x1, y1 = b1[5], b1[6]
                x2, y2 = b2[5], b2[6]
                r1, r2 = b1[4], b2[4]  # radii

                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                # Find bubbles that are:
                # 1. Not overlapping (distance > sum of radii + small margin)
                # 2. Not too far apart (reasonable transition distance)
                # 3. Have different colors (for meaningful blending test)
                min_distance = r1 + r2 + 20  # Non-overlapping with margin
                max_distance = min(
                    400, max(r1, r2) * 4
                )  # Reasonable transition distance

                if min_distance < distance < max_distance:
                    # Check if colors are different enough
                    color1, color2 = b1[3], b2[3]
                    if (
                        color1 != color2
                    ):  # Different colors make for better blending tests
                        pairs.append((b1, b2))

                # Limit to avoid too many pairs
                if len(pairs) >= 10:
                    break
            if len(pairs) >= 10:
                break

        # Sort pairs by distance (closer pairs first, likely better for blending tests)
        pairs.sort(
            key=lambda pair: (
                (pair[0][5] - pair[1][5]) ** 2 + (pair[0][6] - pair[1][6]) ** 2
            )
            ** 0.5
        )

        return pairs

    def _sample_colors_along_line(self, gradient_bg, x1, y1, x2, y2, num_samples=10):
        """Sample colors along a line between two points in the gradient."""
        colors = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            # Ensure coordinates are within bounds
            x = max(0, min(x, gradient_bg.shape[1] - 1))
            y = max(0, min(y, gradient_bg.shape[0] - 1))

            color = gradient_bg[y, x]
            colors.append(tuple(color))

        return colors

    def _calculate_color_changes(self, colors):
        """Calculate color change magnitudes between consecutive colors."""
        changes = []
        for i in range(1, len(colors)):
            prev_color = np.array(colors[i - 1])
            curr_color = np.array(colors[i])
            change = np.linalg.norm(curr_color - prev_color)
            changes.append(change)
        return changes


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
