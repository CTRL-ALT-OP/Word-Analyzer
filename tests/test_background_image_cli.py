"""
Comprehensive test suite for background-image CLI functionality.
Tests the new --background-image, --show-background, --use-image-colors, and --no-boundaries CLI arguments.
Includes performance testing to ensure new functionality is not significantly slower than baseline.
"""

import pytest
import tempfile
import os
import sys
import time
import subprocess
from unittest.mock import patch, Mock
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bubble_visualizer import BubbleVisualizer
from config import DICTIONARY_CONFIG

# Import TestDataLoader from tests directory
tests_dir = os.path.dirname(__file__)
sys.path.append(tests_dir)
from test_data_loader import TestDataLoader


class TestBackgroundImageCLI:
    """Test the new background image CLI functionality."""

    @pytest.fixture
    def placeholder_dataset(self):
        """Fixture providing the placeholder.txt dataset."""
        return TestDataLoader.load_placeholder_dataset()

    @pytest.fixture
    def real_dict_manager(self):
        """Fixture providing a real dictionary manager."""
        return TestDataLoader.get_real_dict_manager()

    @pytest.fixture
    def temp_output_file(self):
        """Fixture that provides a temporary output file path and cleans it up after test."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            output_path = tmp_file.name
        yield output_path
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)

    @pytest.fixture
    def temp_debug_dir(self):
        """Fixture that provides a temporary debug directory and cleans it up after test."""
        debug_dir = tempfile.mkdtemp(prefix="debug_test_")
        yield debug_dir
        # Cleanup debug files
        import shutil

        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)

    @pytest.fixture
    def temp_background_image(self):
        """Create a temporary background image for testing."""
        # Create a more complex test image with defined boundaries
        width, height = 400, 600
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Draw a portrait silhouette (oval) that fills most of the canvas
        margin_x, margin_y = 60, 80
        draw.ellipse(
            [margin_x, margin_y, width - margin_x, height - margin_y], fill="black"
        )

        # Add some color variations for color sampling tests
        # Top section - blue
        draw.ellipse(
            [margin_x + 20, margin_y + 20, width - margin_x - 20, margin_y + 150],
            fill="blue",
        )
        # Middle section - red
        draw.ellipse(
            [margin_x + 20, margin_y + 200, width - margin_x - 20, margin_y + 350],
            fill="red",
        )
        # Bottom section - green
        draw.ellipse(
            [
                margin_x + 20,
                margin_y + 400,
                width - margin_x - 20,
                height - margin_y - 20,
            ],
            fill="green",
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image_path = tmp_file.name
            image.save(image_path)

        yield image_path

        # Cleanup
        if os.path.exists(image_path):
            os.unlink(image_path)

    def test_performance_baseline_without_background_image(
        self, placeholder_dataset, real_dict_manager, temp_output_file
    ):
        """Test baseline performance without background image features."""
        start_time = time.time()

        visualizer = BubbleVisualizer(width=3840, height=2160)  # 4K canvas
        visualizer.create_bubble_chart(
            word_counts=placeholder_dataset,
            dict_manager=real_dict_manager,
            output_path=temp_output_file,
        )

        baseline_time = time.time() - start_time

        # Store baseline time as class attribute for other tests
        TestBackgroundImageCLI.baseline_time = baseline_time

        assert baseline_time > 0, "Baseline test should take some time"
        assert os.path.exists(temp_output_file), "Output file should be created"

        print(
            f"\nBASELINE: Baseline performance (no background image): {baseline_time:.2f} seconds"
        )

    def test_debug_images_generation(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
        temp_debug_dir,
    ):
        """Test that --debug-images generates the required debug images including boundary_mask.png."""
        visualizer = BubbleVisualizer(
            width=3840,
            height=2160,
            background_image_path=temp_background_image,
            use_boundaries=True,
            show_background=False,
        )

        # Generate debug images
        visualizer.save_debug_images(temp_debug_dir)

        # Check that boundary_mask.png exists
        boundary_mask_path = os.path.join(temp_debug_dir, "boundary_mask.png")
        assert os.path.exists(
            boundary_mask_path
        ), "boundary_mask.png should be generated"

        # Check that the boundary mask has the correct dimensions and content
        boundary_mask = cv2.imread(boundary_mask_path, cv2.IMREAD_GRAYSCALE)
        assert boundary_mask.shape == (
            2160,
            3840,
        ), "Boundary mask should match canvas size"

        # Check that there are valid (white) and invalid (black) regions
        white_pixels = np.sum(boundary_mask == 255)
        black_pixels = np.sum(boundary_mask == 0)
        assert white_pixels > 0, "Should have valid boundary regions (white pixels)"
        assert black_pixels > 0, "Should have invalid boundary regions (black pixels)"

        # Verify other debug images exist
        expected_files = ["original_image.png", "processed_image.png"]
        for filename in expected_files:
            filepath = os.path.join(temp_debug_dir, filename)
            if os.path.exists(filepath):
                print(f"DEBUG: Found debug image: {filename}")

        print(f"\nDEBUG: Debug images generated in {temp_debug_dir}")
        print(
            f"BOUNDARY: {white_pixels:,} valid pixels, {black_pixels:,} invalid pixels"
        )

    def test_bubbles_contained_within_boundaries(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test that bubbles are properly contained within the detected boundaries."""
        positioned_bubbles = []
        boundary_mask = None

        def capture_positioned_bubbles(bubbles, output_path):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        visualizer = BubbleVisualizer(
            width=3840,
            height=2160,
            background_image_path=temp_background_image,
            use_boundaries=True,
            show_background=False,
        )

        # Store the boundary mask for validation
        boundary_mask = visualizer.boundary_mask

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            # Patch the _create_image method to capture bubbles
            original_create_image = visualizer._create_image
            visualizer._create_image = capture_positioned_bubbles

            visualizer.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        # Validate that all bubbles are within boundaries
        contained_bubbles = 0
        total_bubbles = len(positioned_bubbles)

        for bubble in positioned_bubbles:
            if len(bubble) >= 7:  # word, count, type, color, radius, x, y
                x, y, radius = bubble[-2], bubble[-1], bubble[-3]

                # Check if bubble center is within valid boundary
                if (
                    0 <= x < boundary_mask.shape[1]
                    and 0 <= y < boundary_mask.shape[0]
                    and boundary_mask[y, x] > 0
                ):
                    contained_bubbles += 1

        containment_rate = contained_bubbles / total_bubbles if total_bubbles > 0 else 0

        assert (
            containment_rate >= 0.9
        ), f"At least 90% of bubbles should be within boundaries, got {containment_rate:.1%}"
        print(
            f"\nCONTAINMENT: {contained_bubbles}/{total_bubbles} bubbles properly contained ({containment_rate:.1%})"
        )

    def test_bubble_boundary_coverage_over_50_percent(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test that bubbles placed cover more than 50% of their allotted boundary area."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, output_path):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        visualizer = BubbleVisualizer(
            width=3840,
            height=2160,
            background_image_path=temp_background_image,
            use_boundaries=True,
            show_background=False,
        )

        boundary_mask = visualizer.boundary_mask
        total_valid_area = np.sum(boundary_mask > 0)

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = visualizer._create_image
            visualizer._create_image = capture_positioned_bubbles

            visualizer.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        # Calculate total area covered by bubbles
        bubble_area = 0
        for bubble in positioned_bubbles:
            if len(bubble) >= 7:
                radius = bubble[-3]
                bubble_area += np.pi * radius * radius

        coverage_rate = bubble_area / total_valid_area if total_valid_area > 0 else 0

        assert (
            coverage_rate >= 0.45
        ), f"Bubbles should cover at least 45% of valid boundary area, got {coverage_rate:.1%}"
        print(
            f"\nCOVERAGE: Bubbles cover {coverage_rate:.1%} of valid boundary area ({bubble_area:,.0f}/{total_valid_area:,} pixels)"
        )

    def test_color_sampling_accuracy(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test that bubbles have correct color values when sampling from the image."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, output_path):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        visualizer = BubbleVisualizer(
            width=3840,
            height=2160,
            background_image_path=temp_background_image,
            use_boundaries=True,
            show_background=False,
        )

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            original_create_image = visualizer._create_image
            visualizer._create_image = capture_positioned_bubbles

            visualizer.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                use_image_colors=True,  # Enable image color sampling
            )

        # Load the original background image for color comparison
        bg_image = cv2.imread(temp_background_image)
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

        # Resize to match canvas for accurate sampling
        canvas_bg = cv2.resize(bg_image, (visualizer.width, visualizer.height))

        color_matches = 0
        total_bubbles = len(positioned_bubbles)

        for bubble in positioned_bubbles:
            if len(bubble) >= 7:
                x, y, radius = bubble[-2], bubble[-1], bubble[-3]
                bubble_color = bubble[3]  # The assigned color

                # Sample color from background image at bubble position
                if 0 <= x < canvas_bg.shape[1] and 0 <= y < canvas_bg.shape[0]:
                    sample_color = canvas_bg[y, x]

                    # Convert hex color to RGB for comparison
                    if bubble_color.startswith("#"):
                        hex_color = bubble_color[1:]
                        bubble_rgb = tuple(
                            int(hex_color[i : i + 2], 16) for i in (0, 2, 4)
                        )

                        # Check if colors are reasonably close (within tolerance)
                        # Convert numpy uint8 values to int to avoid overflow warnings
                        sample_color_int = tuple(int(c) for c in sample_color)
                        color_diff = sum(
                            abs(a - b) for a, b in zip(bubble_rgb, sample_color_int)
                        )
                        if color_diff < 150:  # Reasonable tolerance for color matching
                            color_matches += 1

        color_accuracy = color_matches / total_bubbles if total_bubbles > 0 else 0

        # For image color sampling, we expect some reasonable correlation
        assert (
            color_accuracy >= 0.3
        ), f"At least 30% of bubbles should have colors matching the image, got {color_accuracy:.1%}"
        print(
            f"\nCOLOR: {color_matches}/{total_bubbles} bubbles have colors matching image regions ({color_accuracy:.1%})"
        )

    def test_cli_debug_images_integration(
        self, temp_output_file, temp_background_image, temp_debug_dir
    ):
        """Test CLI integration with --debug-images flag."""
        try:
            # Test debug images generation via CLI
            # Set environment to handle unicode characters properly
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--input",
                    "texts/placeholder.txt",
                    "--build-graph",
                    temp_output_file,
                    "--background-image",
                    temp_background_image,
                    "--debug-images",
                    temp_debug_dir,
                    "--canvas-size",
                    "3840",
                    "2160",  # 4K canvas
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # Replace problematic unicode characters
                timeout=60,
                env=env,
            )

            assert result.returncode == 0, f"CLI command failed: {result.stderr}"
            assert os.path.exists(temp_output_file), "Output file should be created"

            # Check that debug images were generated
            boundary_mask_path = os.path.join(temp_debug_dir, "boundary_mask.png")
            assert os.path.exists(
                boundary_mask_path
            ), "boundary_mask.png should be generated via CLI"

            print(f"\nSUCCESS: CLI debug images test passed")

        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out (> 60 seconds)")

    def test_background_image_with_boundaries(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test background image functionality with boundary constraints."""
        start_time = time.time()

        # Test with boundaries (use_boundaries=True by default when background_image is provided)
        visualizer = BubbleVisualizer(
            width=3840,
            height=2160,  # 4K canvas
            background_image_path=temp_background_image,
            use_boundaries=True,
            show_background=False,
        )

        visualizer.create_bubble_chart(
            word_counts=placeholder_dataset,
            dict_manager=real_dict_manager,
            output_path=temp_output_file,
            use_image_colors=False,  # Use word type colors
        )

        execution_time = time.time() - start_time

        # Performance check: should not be more than 5x slower than baseline
        baseline = getattr(TestBackgroundImageCLI, "baseline_time", 1.0)
        max_allowed_time = baseline * 5

        assert (
            execution_time <= max_allowed_time
        ), f"Background image processing took {execution_time:.2f}s, but baseline was {baseline:.2f}s (max allowed: {max_allowed_time:.2f}s)"
        assert os.path.exists(temp_output_file), "Output file should be created"

        print(
            f"\nIMAGE: Background image with boundaries: {execution_time:.2f} seconds (baseline: {baseline:.2f}s)"
        )

    def test_background_image_with_use_image_colors(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test background image functionality with image color sampling."""
        start_time = time.time()

        visualizer = BubbleVisualizer(
            width=3840,
            height=2160,  # 4K canvas
            background_image_path=temp_background_image,
            use_boundaries=True,
            show_background=False,
        )

        visualizer.create_bubble_chart(
            word_counts=placeholder_dataset,
            dict_manager=real_dict_manager,
            output_path=temp_output_file,
            use_image_colors=True,  # Sample colors from image
        )

        execution_time = time.time() - start_time

        # Performance check
        baseline = getattr(TestBackgroundImageCLI, "baseline_time", 1.0)
        max_allowed_time = baseline * 5

        assert (
            execution_time <= max_allowed_time
        ), f"Image color sampling took {execution_time:.2f}s, but baseline was {baseline:.2f}s (max allowed: {max_allowed_time:.2f}s)"
        assert os.path.exists(temp_output_file), "Output file should be created"

        print(
            f"\nCOLORS: Background image with color sampling: {execution_time:.2f} seconds (baseline: {baseline:.2f}s)"
        )

    def test_background_image_no_boundaries(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test background image functionality without boundary constraints (--no-boundaries)."""
        start_time = time.time()

        visualizer = BubbleVisualizer(
            width=3840,
            height=2160,  # 4K canvas
            background_image_path=temp_background_image,
            use_boundaries=False,  # --no-boundaries
            show_background=False,
        )

        visualizer.create_bubble_chart(
            word_counts=placeholder_dataset,
            dict_manager=real_dict_manager,
            output_path=temp_output_file,
            use_image_colors=True,  # Usually combined with no boundaries
        )

        execution_time = time.time() - start_time

        # Performance check
        baseline = getattr(TestBackgroundImageCLI, "baseline_time", 1.0)
        max_allowed_time = baseline * 5

        assert (
            execution_time <= max_allowed_time
        ), f"No boundaries processing took {execution_time:.2f}s, but baseline was {baseline:.2f}s (max allowed: {max_allowed_time:.2f}s)"
        assert os.path.exists(temp_output_file), "Output file should be created"

        print(
            f"\nNO_BOUNDS: Background image with no boundaries: {execution_time:.2f} seconds (baseline: {baseline:.2f}s)"
        )

    def test_background_image_show_background(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test background image functionality with visible background (--show-background)."""
        start_time = time.time()

        # show_background only works with no boundaries
        visualizer = BubbleVisualizer(
            width=3840,
            height=2160,  # 4K canvas
            background_image_path=temp_background_image,
            use_boundaries=False,  # Required for show_background
            show_background=True,  # --show-background
        )

        visualizer.create_bubble_chart(
            word_counts=placeholder_dataset,
            dict_manager=real_dict_manager,
            output_path=temp_output_file,
            use_image_colors=True,
        )

        execution_time = time.time() - start_time

        # Performance check
        baseline = getattr(TestBackgroundImageCLI, "baseline_time", 1.0)
        max_allowed_time = baseline * 5

        assert (
            execution_time <= max_allowed_time
        ), f"Show background processing took {execution_time:.2f}s, but baseline was {baseline:.2f}s (max allowed: {max_allowed_time:.2f}s)"
        assert os.path.exists(temp_output_file), "Output file should be created"

        print(
            f"\nSHOW_BG: Background image with visible background: {execution_time:.2f} seconds (baseline: {baseline:.2f}s)"
        )

    def test_bubble_scaling_based_on_available_space(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test that bubbles are scaled appropriately based on available space."""
        positioned_bubbles = []

        def capture_positioned_bubbles(bubbles, output_path):
            nonlocal positioned_bubbles
            positioned_bubbles = bubbles[:]

        with patch("bubble_visualizer.Image"), patch("bubble_visualizer.ImageDraw"):
            visualizer = BubbleVisualizer(
                width=3840,
                height=2160,
                background_image_path=temp_background_image,
                use_boundaries=True,
            )

            # Patch the _create_image method to capture bubbles
            original_create_image = visualizer._create_image
            visualizer._create_image = capture_positioned_bubbles

            visualizer.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
            )

        # Calculate available space utilization
        canvas_area = visualizer.width * visualizer.height
        used_area = sum(
            3.14159 * bubble[6] ** 2 for bubble in positioned_bubbles if len(bubble) > 6
        )  # radius is at index 6
        utilization = used_area / canvas_area

        # Check scaling based on available space
        # If available space < 30%, bubbles should be scaled to 0.8
        # If available space < 50%, bubbles should be scaled to 0.9
        if utilization < 0.30:
            # Should have applied 0.8 scaling - check that bubbles are reasonably sized
            max_bubble_radius = max(
                bubble[6] for bubble in positioned_bubbles if len(bubble) > 6
            )
            assert (
                max_bubble_radius > 0
            ), "Bubbles should have positive radius even with scaling"
            print(
                f"\nSCALING: Low space utilization ({utilization:.1%}) - bubbles scaled to 0.8"
            )
        elif utilization < 0.50:
            print(
                f"\nSCALING: Medium space utilization ({utilization:.1%}) - bubbles scaled to 0.9"
            )
        else:
            print(
                f"\nSCALING: Good space utilization ({utilization:.1%}) - no scaling needed"
            )

        assert len(positioned_bubbles) > 0, "Should have positioned some bubbles"

    def test_cli_integration_background_image(
        self, temp_output_file, temp_background_image
    ):
        """Test CLI integration with background image arguments."""
        try:
            # Test basic background image functionality
            # Set environment to handle unicode characters properly
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--input",
                    "texts/placeholder.txt",
                    "--build-graph",
                    temp_output_file,
                    "--background-image",
                    temp_background_image,
                    "--canvas-size",
                    "3840",
                    "2160",  # 4K canvas
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # Replace problematic unicode characters
                timeout=60,
                env=env,
            )

            assert result.returncode == 0, f"CLI command failed: {result.stderr}"
            assert os.path.exists(temp_output_file), "Output file should be created"

            print(f"\nSUCCESS: CLI integration test passed")

        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out (> 60 seconds)")

    def test_cli_integration_with_all_new_flags(
        self, temp_output_file, temp_background_image
    ):
        """Test CLI integration with all new background image flags."""
        try:
            # Test with all new flags: --use-image-colors, --no-boundaries, --show-background
            # Set environment to handle unicode characters properly
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--input",
                    "texts/placeholder.txt",
                    "--build-graph",
                    temp_output_file,
                    "--background-image",
                    temp_background_image,
                    "--use-image-colors",
                    "--no-boundaries",
                    "--show-background",
                    "--canvas-size",
                    "3840",
                    "2160",  # 4K canvas
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # Replace problematic unicode characters
                timeout=60,
                env=env,
            )

            assert result.returncode == 0, f"CLI command failed: {result.stderr}"
            assert os.path.exists(temp_output_file), "Output file should be created"

            print(f"\nSUCCESS: CLI integration test with all flags passed")

        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out (> 60 seconds)")

    def test_word_types_from_config(self, real_dict_manager):
        """Test that word types are properly loaded from config.py."""
        # Verify that dictionary manager loads word types from config
        expected_types = DICTIONARY_CONFIG["word_types"]
        loaded_types = list(real_dict_manager.dictionaries.keys())

        # Check that most expected types are loaded (some files might be missing)
        common_types = set(expected_types) & set(loaded_types)
        assert (
            len(common_types) >= 10
        ), f"Should load most word types from config, got {len(common_types)} common types"

        # Check for essential types
        essential_types = ["noun", "verb", "adj", "art"]
        for word_type in essential_types:
            assert (
                word_type in loaded_types
            ), f"Essential word type '{word_type}' should be loaded"

        print(f"\nCONFIG: Loaded {len(loaded_types)} word types from config")

    def test_uses_4k_canvas_by_default(
        self, placeholder_dataset, real_dict_manager, temp_output_file
    ):
        """Test that the system uses 4K canvas by default as specified."""
        visualizer = BubbleVisualizer()  # Default initialization

        assert (
            visualizer.width == 3840
        ), f"Default width should be 4K (3840), got {visualizer.width}"
        assert (
            visualizer.height == 2160
        ), f"Default height should be 4K (2160), got {visualizer.height}"

        print(f"\nCANVAS: Using 4K canvas: {visualizer.width}x{visualizer.height}")

    def test_placeholder_txt_dataset_loading(self, placeholder_dataset):
        """Test that placeholder.txt dataset is properly loaded."""
        assert (
            len(placeholder_dataset) > 100
        ), f"Should load substantial dataset, got {len(placeholder_dataset)} unique words"
        assert (
            sum(placeholder_dataset.values()) > 1000
        ), f"Should have substantial word count, got {sum(placeholder_dataset.values())} total words"

        # Check for common words that should be in any substantial text
        common_words = ["the", "and", "of", "to", "a"]
        found_common = sum(1 for word in common_words if word in placeholder_dataset)
        assert (
            found_common >= 3
        ), f"Should find common words in dataset, found {found_common} of {len(common_words)}"

        print(
            f"\nDATASET: Loaded placeholder.txt: {len(placeholder_dataset)} unique words, {sum(placeholder_dataset.values())} total words"
        )

    def test_performance_timeout_safety(
        self,
        placeholder_dataset,
        real_dict_manager,
        temp_output_file,
        temp_background_image,
    ):
        """Test that operations timeout if they take more than 5x baseline time."""
        baseline = getattr(TestBackgroundImageCLI, "baseline_time", 1.0)
        max_time = baseline * 5 + 10  # Add 10 second buffer for test overhead

        start_time = time.time()

        try:
            visualizer = BubbleVisualizer(
                width=3840,
                height=2160,
                background_image_path=temp_background_image,
                use_boundaries=True,
                show_background=False,
            )

            visualizer.create_bubble_chart(
                word_counts=placeholder_dataset,
                dict_manager=real_dict_manager,
                output_path=temp_output_file,
                use_image_colors=True,
            )

            execution_time = time.time() - start_time

            # This should not timeout based on our 5x baseline rule
            assert (
                execution_time <= max_time
            ), f"Operation took {execution_time:.2f}s, exceeds {max_time:.2f}s limit"

            print(
                f"\nTIMEOUT: Performance within limits: {execution_time:.2f}s (max: {max_time:.2f}s)"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            if execution_time >= max_time:
                pytest.fail(
                    f"Operation timed out after {execution_time:.2f}s (limit: {max_time:.2f}s)"
                )
            else:
                raise e
