"""
Bubble Chart Visualizer for Word Analysis
Creates bubble charts with words sized by frequency and colored by word type.
Now supports image-based boundaries and color sampling.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Counter, Optional
from collections import defaultdict

try:
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.colors as mcolors

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from config import DICTIONARY_CONFIG


class BubbleVisualizer:
    """Creates bubble chart visualizations of word frequency data with optional image boundaries."""

    # 4K dimensions (standard)
    DEFAULT_WIDTH = 3840
    DEFAULT_HEIGHT = 2160

    # Color palette for different word types
    WORD_TYPE_COLORS = {
        "conj": "#FF6B6B",  # Red - Conjunction
        "art": "#4ECDC4",  # Teal - Article
        "adj": "#45B7D1",  # Blue - Adjective
        "adv": "#96CEB4",  # Green - Adverb
        "prep": "#FFEAA7",  # Yellow - Preposition
        "noun": "#DDA0DD",  # Plum - Noun
        "verb": "#98D8C8",  # Mint - Verb
        "dpron": "#F7DC6F",  # Light Yellow - Demonstrative pronoun
        "indpron": "#BB8FCE",  # Light Purple - Indefinite pronoun
        "intpron": "#85C1E9",  # Light Blue - Interrogative pronoun
        "opron": "#F8C471",  # Orange - Other pronoun
        "ppron": "#82E0AA",  # Light Green - Personal pronoun
        "refpron": "#F1948A",  # Light Red - Reflexive pronoun
        "relpron": "#AED6F1",  # Very Light Blue - Relative pronoun
        "spron": "#D2B4DE",  # Light Lavender - Subject pronoun
        "pnoun": "#F5B041",  # Gold - Proper noun
        "unknown": "#BDC3C7",  # Gray - Unknown type
    }

    def __init__(
        self,
        width: int = None,
        height: int = None,
        background_image_path: str = None,
        use_boundaries: bool = True,
        show_background: bool = False,
    ):
        """Initialize the bubble visualizer with optional background image."""
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL (Pillow) is required for bubble visualization. Install with: pip install Pillow matplotlib"
            )

        if background_image_path and not OPENCV_AVAILABLE:
            raise ImportError(
                "OpenCV is required for image processing features. Install with: pip install opencv-python"
            )

        self.width = width or self.DEFAULT_WIDTH
        self.height = height or self.DEFAULT_HEIGHT
        self.bubbles = []  # List of (x, y, radius, word, color) tuples

        # Image processing attributes
        self.background_image_path = background_image_path
        self.background_image = None
        self.processed_image = None
        self.boundary_mask = None
        self.valid_positions = None
        self.use_boundaries = (
            use_boundaries  # Whether to use image boundaries for placement
        )
        self.show_background = (
            show_background  # Whether to display background image in final chart
        )

        # Performance optimization: cache valid position coordinates
        self._valid_coords_cache = None

        # Process image if provided
        if background_image_path:
            self._process_background_image()

    def _process_background_image(self) -> None:
        """Process the background image to create boundaries and prepare for color sampling."""
        if not self.background_image_path:
            return

        print(f"Processing background image: {self.background_image_path}")

        # Load image
        try:
            original_image = cv2.imread(self.background_image_path)
            if original_image is None:
                raise ValueError(f"Could not load image: {self.background_image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # Remove background from original high-resolution image (better quality)
        print("Processing background removal on original image...")
        if self.use_boundaries:
            self._remove_background_from_original(original_image)
        else:
            self.processed_image = original_image
            self.boundary_mask = None
            self.valid_positions = None
            self._valid_coords_cache = None
            print("Image loaded for color sampling only - boundaries ignored")

        # Now resize both the processed image and boundary mask to canvas size
        self.background_image = self._resize_image_preserve_aspect(self.processed_image)

        if self.use_boundaries:
            self.boundary_mask = self._resize_mask_preserve_aspect(
                self.boundary_mask, original_image.shape[:2]
            )
            # Create valid position mask for bubble placement
            self._create_valid_positions_mask()
        else:
            # Not using boundaries - set to None to allow full canvas placement
            self.boundary_mask = None
            self.valid_positions = None
            self._valid_coords_cache = None
            print("Image loaded for color sampling only - boundaries ignored")

        # Update processed image reference
        self.processed_image = self.background_image

        print("Background image processing complete")

    def _remove_background_from_original(self, original_image) -> None:
        """Remove background from the original high-resolution image for better quality."""
        print("Removing background...")

        # Method 1: GrabCut algorithm (works well for portraits/objects)
        mask = np.zeros(original_image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Define rectangle around the presumed foreground (center 80% of image)
        height, width = original_image.shape[:2]
        rect = (
            int(width * 0.1),
            int(height * 0.1),
            int(width * 0.8),
            int(height * 0.8),
        )

        try:
            cv2.grabCut(
                original_image,
                mask,
                rect,
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_RECT,
            )

            # Create binary mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

            # Apply Gaussian blur to smooth edges
            mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)

            self.boundary_mask = mask2

            # Create processed image (for color sampling)
            self.processed_image = original_image.copy()

        except Exception as e:
            print(f"GrabCut failed: {e}, trying alternative method...")
            self._remove_background_alternative_original(original_image)

    def _remove_background_alternative_original(self, original_image) -> None:
        """Alternative background removal using edge detection and contours on original image."""
        print("Using alternative background removal method...")

        # Convert to grayscale
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive threshold to create binary image
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create mask from largest contour (assumed to be main subject)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)

            # Normalize to 0-1 range
            self.boundary_mask = mask.astype("uint8") // 255
        else:
            # Fallback: use center region
            mask = np.zeros(gray.shape, np.uint8)
            h, w = mask.shape
            cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 2), 0, 0, 360, 1, -1)
            self.boundary_mask = mask

        self.processed_image = original_image.copy()

    def _resize_mask_preserve_aspect(self, mask, original_shape):
        """Resize boundary mask to match the resized image canvas."""
        original_height, original_width = original_shape

        # Calculate scale factor to fit within canvas (same as image resizing)
        scale_w = self.width / original_width
        scale_h = self.height / original_height
        scale = min(scale_w, scale_h)

        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize mask
        resized_mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height))

        # Create canvas and center the mask
        canvas_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Calculate positioning to center the mask
        start_y = (self.height - new_height) // 2
        start_x = (self.width - new_width) // 2

        # Place resized mask on canvas
        canvas_mask[start_y : start_y + new_height, start_x : start_x + new_width] = (
            resized_mask
        )

        return canvas_mask

    def _create_valid_positions_mask(self) -> None:
        """Create a mask of valid positions for bubble placement."""
        if self.boundary_mask is None:
            return

        print("Creating valid positions mask...")

        # Use boundary mask directly instead of heavy erosion
        # This gives much better results and avoids the "wild" appearance
        self.valid_positions = self.boundary_mask.astype(np.uint8)

        # Cache valid coordinates for performance optimization
        self._cache_valid_coordinates()

        # Count valid pixels for statistics
        valid_pixels = np.sum(self.valid_positions)
        total_pixels = self.width * self.height
        coverage = valid_pixels / total_pixels

        print(
            f"Valid placement area: {coverage:.1%} of total canvas ({valid_pixels:,} pixels)"
        )

        # Provide scaling guidance
        if coverage < 0.2:
            print(
                "WARNING: Very constrained space - bubbles will be significantly smaller"
            )
        elif coverage < 0.4:
            print("WARNING: Limited space - bubbles will be moderately smaller")
        elif coverage < 0.6:
            print("INFO: Moderate space - bubbles will be slightly smaller")
        else:
            print("GOOD: Good space available - minimal bubble size reduction")

    def _cache_valid_coordinates(self) -> None:
        """Cache valid coordinates for fast sampling - major performance optimization."""
        if self.valid_positions is None:
            self._valid_coords_cache = None
            return

        # Get all valid coordinates once and cache them
        y_coords, x_coords = np.where(self.valid_positions > 0)
        if len(x_coords) > 0:
            # Store as list of tuples for fast random access
            self._valid_coords_cache = list(zip(x_coords, y_coords))
            print(
                f"Cached {len(self._valid_coords_cache)} valid positions for fast sampling"
            )
        else:
            self._valid_coords_cache = [(self.width // 2, self.height // 2)]  # Fallback
            print("No valid positions found, using center as fallback")

    def _is_position_valid(self, x: int, y: int, radius: int) -> bool:
        """Check if a position is valid for bubble placement within image boundaries."""
        if self.valid_positions is None:
            # No image boundaries, use default bounds checking
            return (
                radius <= x < self.width - radius and radius <= y < self.height - radius
            )

        # Check bounds first (faster)
        if (
            x - radius < 0
            or x + radius >= self.width
            or y - radius < 0
            or y + radius >= self.height
        ):
            return False

            # For boundary mask, check key points around the circle
        # Use even fewer samples during fast testing for better performance
        num_samples = max(3, radius // 15)  # Very minimal sampling for speed

        # Quick center check first - if center is invalid, position is definitely bad
        if self.valid_positions[y, x] == 0:
            return False

        # Sample fewer points around perimeter
        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            check_x = int(
                x + radius * 0.7 * math.cos(angle)
            )  # Check slightly inside radius (more lenient)
            check_y = int(y + radius * 0.7 * math.sin(angle))

            # Check if position is in valid area
            if self.valid_positions[check_y, check_x] == 0:
                return False

        return True

    def _sample_color_from_image(self, x: int, y: int, radius: int) -> str:
        """Sample average color from the image at bubble location."""
        if self.processed_image is None:
            return self.WORD_TYPE_COLORS["unknown"]

        # Create sampling area (smaller than bubble to avoid edges)
        sample_radius = max(1, radius // 3)

        # Get bounding box for sampling
        x1 = max(0, x - sample_radius)
        y1 = max(0, y - sample_radius)
        x2 = min(self.width, x + sample_radius)
        y2 = min(self.height, y + sample_radius)

        # Extract region and calculate average color
        region = self.processed_image[y1:y2, x1:x2]
        if region.size > 0:
            # Calculate average color (BGR to RGB)
            avg_color = np.mean(region, axis=(0, 1))
            # Convert BGR to RGB and to hex
            r, g, b = int(avg_color[2]), int(avg_color[1]), int(avg_color[0])
            return f"#{r:02x}{g:02x}{b:02x}"

        return self.WORD_TYPE_COLORS["unknown"]

    def create_bubble_chart(
        self,
        word_counts: Counter,
        dict_manager,
        output_path: str,
        min_bubble_size: int = 2,
        exclude_types: Optional[List[str]] = None,
        use_image_colors: bool = False,
        gradient_mode: bool = False,
    ) -> None:
        """
        Create a bubble chart from word frequency data.

        Args:
            word_counts: Counter of word frequencies
            dict_manager: DictionaryManager for word type lookup
            output_path: Path to save the image
            min_bubble_size: Minimum bubble size in pixels
            exclude_types: List of word types to exclude (e.g., ["art", "conj", "prep"])
            use_image_colors: If True and background image is provided, sample colors from image
            gradient_mode: If True, create smooth gradient transitions between bubbles instead of hard circles
        """
        # Filter out words from excluded types
        if exclude_types:
            word_counts = self._filter_excluded_types(
                word_counts, dict_manager, exclude_types
            )

        # Filter out words that would be too small
        filtered_words = self._filter_small_words(word_counts, min_bubble_size)
        if not filtered_words:
            print("No words large enough to display (all would be < 1px)")
            return

        # Get word types and colors
        word_data = self._prepare_word_data(filtered_words, dict_manager)
        print("Word data prepared")

        # Calculate bubble sizes
        print("Calculating bubble sizes...")
        bubble_data = self._calculate_bubble_sizes(word_data)
        print("Bubble size data calculated")

        # Position bubbles (avoid overlap and respect image boundaries)
        print("Positioning bubbles...")
        positioned_bubbles = self._position_bubbles(bubble_data, use_image_colors)
        print("Bubbles positioned")

        # Create and save image
        self._create_image(positioned_bubbles, output_path, gradient_mode=gradient_mode)

        excluded_info = ""
        if exclude_types:
            excluded_info = f" (excluded types: {', '.join(exclude_types)})"

        image_info = ""
        if self.background_image_path:
            color_info = " with image colors" if use_image_colors else ""
            image_info = f" using image boundaries{color_info}"

        gradient_info = " with gradient transitions" if gradient_mode else ""

        print(f"Bubble chart saved to: {output_path}")
        print(
            f"Generated {len(positioned_bubbles)} bubbles from {len(word_counts)} words{excluded_info}{image_info}{gradient_info}"
        )

    def _filter_small_words(self, word_counts: Counter, min_size: int) -> Counter:
        """
        Find optimal frequency cutoff using iterative testing to ensure all selected words can be placed.
        This prevents the problem of skipping important words while placing less important ones.
        """
        if not word_counts:
            return Counter()

        frequencies = sorted(word_counts.values(), reverse=True)
        if len(frequencies) <= 50:
            return word_counts  # Small datasets - try to place everything

        print(
            f"Finding optimal cutoff for {len(frequencies)} words (range {min(frequencies)}-{max(frequencies)})"
        )

        # More aggressive search - test many cutoff values systematically
        best_cutoff = 1
        best_count = 0
        best_ratio = 0

        # Start with a broader survey of potential cutoffs
        max_freq = max(frequencies)
        min_freq = min(frequencies)

        # Strategy 1: Test key frequency levels that appear in the data
        unique_frequencies = sorted(set(frequencies), reverse=True)
        test_cutoffs = list(unique_frequencies[:50])
        # Strategy 2: Test logarithmic intervals for broader coverage
        for i in range(100):  # More granular steps
            if max_freq > min_freq:
                # Logarithmic spacing to cover the full range
                log_min = math.log(min_freq) if min_freq > 0 else 0
                log_max = math.log(max_freq)
                log_cutoff = log_min + (log_max - log_min) * (1 - i / 99)
                cutoff = max(1, int(math.exp(log_cutoff)))
                test_cutoffs.append(cutoff)

        # Strategy 3: Linear intervals in different ranges
        test_cutoffs.extend(iter(range(1, min(40, max_freq + 1))))
        # Remove duplicates and sort
        test_cutoffs = sorted(set(test_cutoffs))

        print(f"  Testing {len(test_cutoffs)} different cutoff values...")

        # Test all cutoffs and find the best one
        for test_cutoff in test_cutoffs:
            # Create test dataset with this cutoff
            test_words = Counter(
                {
                    word: count
                    for word, count in word_counts.items()
                    if count >= test_cutoff
                }
            )

            if len(test_words) == 0:
                continue

            # Test if this set of words can actually be placed
            placeable_count = self._test_placement_capacity(test_words)
            success_rate = placeable_count / len(test_words)

            # For image boundaries, be more aggressive about accepting lower success rates
            # since the boundary constraints naturally limit placement
            min_success_rate = 0.60 if self.valid_positions is not None else 0.75

            # Calculate a score that balances word count and success rate
            # Prefer higher word counts, but require reasonable success rate
            if success_rate >= min_success_rate:
                score = (
                    placeable_count * success_rate
                )  # Favor both high count and high success rate

                if score > best_ratio or (
                    score == best_ratio and len(test_words) > best_count
                ):
                    best_cutoff = test_cutoff
                    best_count = placeable_count
                    best_ratio = score
                    print(
                        f"  New best - Cutoff {test_cutoff}: {len(test_words)} words -> {placeable_count} placeable ({success_rate:.1%}, score: {score:.1f})"
                    )
            elif len(test_cutoffs) <= 20:  # Show details for smaller searches
                print(
                    f"  Cutoff {test_cutoff}: {len(test_words)} words -> {placeable_count} placeable ({success_rate:.1%}) - below threshold ({min_success_rate:.0%})"
                )

        # Use the best cutoff found
        final_words = Counter(
            {word: count for word, count in word_counts.items() if count >= best_cutoff}
        )

        # Post-processing: try to add more words to fill remaining space
        if best_cutoff > 1:
            if remaining_words := Counter(
                {
                    word: count
                    for word, count in word_counts.items()
                    if count < best_cutoff
                    and count >= max(1, best_cutoff - 5)  # Try 5 frequency levels below
                }
            ):
                # Test adding some of these words
                test_expanded = final_words + remaining_words
                expanded_placeable = self._test_placement_capacity(test_expanded)
                expanded_rate = expanded_placeable / len(test_expanded)

                # Use the same threshold as in main cutoff selection
                post_process_threshold = (
                    0.60 if self.valid_positions is not None else 0.75
                )
                if expanded_rate >= post_process_threshold:
                    final_words = test_expanded
                    print(
                        f"  Post-processing: expanded to {len(final_words)} words ({expanded_rate:.1%} estimated success)"
                    )
                else:
                    # If full expansion doesn't work, try smaller increments
                    for freq_level in range(
                        best_cutoff - 1, max(0, best_cutoff - 5), -1
                    ):
                        if subset_words := Counter(
                            {
                                word: count
                                for word, count in word_counts.items()
                                if count >= freq_level and count < best_cutoff
                            }
                        ):
                            test_subset = final_words + subset_words
                            subset_placeable = self._test_placement_capacity(
                                test_subset
                            )
                            subset_rate = subset_placeable / len(test_subset)

                            if subset_rate >= post_process_threshold:
                                final_words = test_subset
                                print(
                                    f"  Post-processing: added frequency {freq_level}+ words -> {len(final_words)} total ({subset_rate:.1%} success)"
                                )
                                break

        print(
            f"Selected cutoff {best_cutoff}: {len(final_words)} words (estimated {best_count} placeable)"
        )
        return final_words

    def _test_placement_capacity(self, word_counts: Counter) -> int:
        """
        Quickly test how many words from this set can actually be placed.
        Uses a simplified/faster placement algorithm for testing.
        """
        if not word_counts:
            return 0

        # Prepare word data (reuse existing method)
        word_data = []
        word_data.extend(
            (word, count, "test", "#000000") for word, count in word_counts.items()
        )
        # Calculate bubble sizes (reuse existing method but simplified)
        bubble_data = self._calculate_bubble_sizes_fast(word_data)

        return self._test_placement_fast(bubble_data)

    def _calculate_bubble_sizes_fast(
        self, word_data: List[Tuple[str, int, str, str]]
    ) -> List[Tuple[str, int, int]]:
        """Fast bubble size calculation for testing - uses same scaling logic as main calculation."""
        if not word_data:
            return []

        max_count = max(count for _, count, _, _ in word_data)
        min_count = min(count for _, count, _, _ in word_data)

        # Use same scaling logic as main bubble calculation
        base_min_radius = 12
        base_max_radius = min(400, min(self.width, self.height) // 6)

        # Adjust for image boundaries if present
        if self.valid_positions is not None:
            valid_area_ratio = np.sum(self.valid_positions) / (self.width * self.height)

            if valid_area_ratio < 0.3:
                size_multiplier = 0.8
            elif valid_area_ratio < 0.5:
                size_multiplier = 0.9
        else:
            size_multiplier = 1.0

        min_radius = max(8, int(base_min_radius * size_multiplier))
        max_radius = int(base_max_radius * size_multiplier)

        bubble_data = []
        for word, count, _, _ in word_data:
            if max_count == min_count:
                radius = max_radius // 2
            else:
                # Simplified scaling
                normalized = (count - min_count) / (max_count - min_count)
                radius = int(min_radius + normalized * (max_radius - min_radius))

            bubble_data.append((word, count, radius))

        return bubble_data

    def _test_placement_fast(self, bubble_data: List[Tuple[str, int, int]]) -> int:
        """
        Fast placement testing - uses grid-based approach for speed.
        Returns number of bubbles that can be placed.
        """
        if not bubble_data:
            return 0

        # Sort by size (largest first)
        sorted_bubbles = sorted(bubble_data, key=lambda x: x[2], reverse=True)

        # Use a grid to track occupied space (much faster than checking all overlaps)
        grid_size = 20  # 20px grid cells
        grid_width = self.width // grid_size + 1
        grid_height = self.height // grid_size + 1
        occupied_grid = [[False] * grid_width for _ in range(grid_height)]

        placed = 0

        # For image boundaries, be more aggressive with attempts and use better strategy
        if self.valid_positions is not None:
            max_attempts = 300  # Much more attempts for image boundaries
            max_no_progress = 50  # Stop if we fail this many bubbles in a row
        else:
            max_attempts = 100
            max_no_progress = 20

        consecutive_failures = 0

        for word, count, radius in sorted_bubbles:
            # Find a position using grid
            found_position = False

            for attempt in range(max_attempts):
                # For image boundaries, prioritize valid coordinate sampling
                if (
                    self.valid_positions is not None
                    and self._valid_coords_cache is not None
                    and len(self._valid_coords_cache) > 0
                ):
                    try:
                        pixel_x, pixel_y = random.choice(self._valid_coords_cache)
                        grid_x = pixel_x // grid_size
                        grid_y = pixel_y // grid_size
                    except Exception:
                        grid_x = random.randint(0, grid_width - 1)
                        grid_y = random.randint(0, grid_height - 1)
                elif attempt < max_attempts // 2:
                    # Try systematic grid positions first
                    grid_x = (
                        attempt * 7
                    ) % grid_width  # Prime number for good distribution
                    grid_y = (attempt * 11) % grid_height
                else:
                    grid_x = random.randint(0, grid_width - 1)
                    grid_y = random.randint(0, grid_height - 1)

                # Convert to pixel coordinates
                pixel_x = grid_x * grid_size + grid_size // 2
                pixel_y = grid_y * grid_size + grid_size // 2

                # Check if position is valid (bounds + image boundaries)
                if not self._is_position_valid(pixel_x, pixel_y, radius):
                    continue

                # Check grid occupancy in radius
                grid_radius = (radius // grid_size) + 1
                collision = False

                for dx in range(-grid_radius, grid_radius + 1):
                    for dy in range(-grid_radius, grid_radius + 1):
                        check_x = grid_x + dx
                        check_y = grid_y + dy

                        if (
                            0 <= check_x < grid_width
                            and 0 <= check_y < grid_height
                            and occupied_grid[check_y][check_x]
                        ):
                            collision = True
                            break
                    if collision:
                        break

                if not collision:
                    # Mark area as occupied
                    for dx in range(-grid_radius, grid_radius + 1):
                        for dy in range(-grid_radius, grid_radius + 1):
                            mark_x = grid_x + dx
                            mark_y = grid_y + dy
                            if 0 <= mark_x < grid_width and 0 <= mark_y < grid_height:
                                occupied_grid[mark_y][mark_x] = True

                    placed += 1
                    found_position = True
                    consecutive_failures = 0  # Reset failure counter
                    break

            if not found_position:
                consecutive_failures += 1
                # If we can't place this bubble and have failed several in a row,
                # we probably can't place smaller ones either in this density
                if consecutive_failures >= max_no_progress:
                    break

        return placed

    def _filter_excluded_types(
        self, word_counts: Counter, dict_manager, exclude_types: List[str]
    ) -> Counter:
        """Filter out words belonging to excluded word types."""
        from config import DICTIONARY_CONFIG

        available_types = DICTIONARY_CONFIG.get("word_types", [])
        excluded_word_types = set()

        # Validate and collect word types to exclude
        for word_type in exclude_types:
            if word_type in available_types:
                excluded_word_types.add(word_type)
            else:
                print(
                    f"Warning: Unknown word type '{word_type}' - available types: {available_types}"
                )

        if not excluded_word_types:
            return word_counts

        # Filter out words of excluded types
        filtered_counts = Counter()
        excluded_count = 0

        for word, count in word_counts.items():
            word_type = dict_manager.get_word_type(word)
            if word_type not in excluded_word_types:
                filtered_counts[word] = count
            else:
                excluded_count += count

        if excluded_count > 0:
            print(
                f"Excluded {excluded_count} words from {len(excluded_word_types)} word types: {sorted(excluded_word_types)}"
            )

        return filtered_counts

    def _prepare_word_data(
        self, word_counts: Counter, dict_manager
    ) -> List[Tuple[str, int, str, str]]:
        """Prepare word data with types and colors."""
        word_data = []

        for word, count in word_counts.items():
            word_type = dict_manager.get_word_type(word) or "unknown"
            color = self.WORD_TYPE_COLORS.get(
                word_type, self.WORD_TYPE_COLORS["unknown"]
            )
            word_data.append((word, count, word_type, color))

        return word_data

    def _calculate_bubble_sizes(
        self, word_data: List[Tuple[str, int, str, str]]
    ) -> List[Tuple[str, int, str, str, int]]:
        """Calculate bubble radii based on word frequency using area-based scaling, adjusted for available space."""
        if not word_data:
            return []

        max_count = max(count for _, count, _, _ in word_data)
        min_count = min(count for _, count, _, _ in word_data)

        # Calculate available area based on image boundaries
        if self.valid_positions is not None:
            # For image boundaries, use actual valid area
            valid_pixels = np.sum(self.valid_positions)
            total_area = 0.6 * valid_pixels  # Use 60% of valid area (more conservative)
            area_ratio = valid_pixels / (self.width * self.height)

            print(f"Valid area: {area_ratio:.1%} of canvas ({valid_pixels:,} pixels)")

            # Adjust scaling based on available space
            if area_ratio < 0.3:  # Very constrained space (< 30% of canvas)
                area_utilization = 0.4  # Use 40% of valid area
                size_multiplier = 0.8  # Smaller bubbles
            elif area_ratio < 0.5:  # Moderately constrained (30-50% of canvas)
                area_utilization = 0.35
                size_multiplier = 0.9
            else:  # Less constrained (> 50% of canvas)
                area_utilization = 0.3
                size_multiplier = 0.9
        else:
            # For full canvas, use traditional scaling
            total_area = 0.8 * self.width * self.height
            area_utilization = 0.3
            size_multiplier = 1.0

        total_frequency = sum(count for _, count, _, _ in word_data)

        # Adjust radius bounds based on available space and canvas size
        base_min_radius = 12
        base_max_radius = min(400, min(self.width, self.height) // 6)

        # Scale radii based on available space
        min_radius = max(8, int(base_min_radius * size_multiplier))
        max_radius = int(base_max_radius * size_multiplier)

        # For very constrained spaces, cap maximum radius more aggressively
        if self.valid_positions is not None:
            valid_area_ratio = np.sum(self.valid_positions) / (self.width * self.height)
            if valid_area_ratio < 0.2:  # Very small valid area
                max_radius = min(max_radius, min(self.width, self.height) // 10)
            elif valid_area_ratio < 0.4:  # Small-medium valid area
                max_radius = min(max_radius, min(self.width, self.height) // 8)

        print(
            f"Bubble size range: {min_radius}-{max_radius} pixels (multiplier: {size_multiplier:.1f})"
        )

        bubble_data = []
        for word, count, word_type, color in word_data:
            if max_count == min_count:
                radius = max_radius // 2
            else:
                # Use square root scaling for area-proportional bubbles
                area_ratio = count / total_frequency
                target_area = area_ratio * total_area * area_utilization

                # Calculate radius from area (area = π * r²)
                radius_from_area = math.sqrt(target_area / math.pi)

                # Also apply logarithmic scaling for better distribution
                log_normalized = (math.log(count) - math.log(min_count)) / (
                    math.log(max_count) - math.log(min_count)
                )
                radius_from_log = min_radius + log_normalized * (
                    max_radius - min_radius
                )

                # Take the geometric mean of both approaches for balanced scaling
                radius = int(math.sqrt(radius_from_area * radius_from_log))
                radius = max(min_radius, min(radius, max_radius))

            bubble_data.append((word, count, word_type, color, radius))

        return bubble_data

    def _position_bubbles(
        self, bubble_data: List[Tuple[str, int, str, str, int]], use_image_colors: bool
    ) -> List[Tuple[str, int, str, str, int, int, int]]:
        """Position bubbles with high success rate, respecting image boundaries if provided."""
        if not bubble_data:
            return []

        # Sort by size (largest first)
        sorted_bubbles = sorted(bubble_data, key=lambda x: x[4], reverse=True)
        positioned = []
        failed = []

        max_attempts = 5000

        for word, count, word_type, color, radius in sorted_bubbles:
            best_position = None
            min_distance_to_edge = float("inf")
            attempts = 0

            # Try multiple strategies with more systematic coverage
            while attempts < max_attempts:
                if attempts < max_attempts // 4:
                    # Strategy 1: Try positions near existing bubbles
                    if positioned:
                        existing_bubble = random.choice(positioned)
                        existing_x, existing_y, existing_radius = (
                            existing_bubble[-2],
                            existing_bubble[-1],
                            existing_bubble[-3],
                        )
                        # Systematic angles around existing bubbles
                        angle = (
                            (attempts * 0.618) * 2 * math.pi
                        )  # Golden ratio for good distribution
                        distance = existing_radius + radius + random.randint(2, 15)
                        x = int(existing_x + distance * math.cos(angle))
                        y = int(existing_y + distance * math.sin(angle))
                    else:
                        # First bubble - try center of valid region
                        x, y = self._get_initial_center_position()

                elif attempts < max_attempts // 2:
                    grid_size = 30
                    # Strategy 2: Sample from valid positions if available
                    if self.valid_positions is None:
                        grid_x = (attempts * 7) % (
                            self.width // grid_size
                        )  # Prime for distribution
                        grid_y = (attempts * 11) % (self.height // grid_size)
                    else:
                        pixel_x, pixel_y = random.choice(self._valid_coords_cache)
                        grid_x = pixel_x // grid_size
                        grid_y = pixel_y // grid_size
                    x = grid_x * grid_size + random.randint(
                        -grid_size // 4, grid_size // 4
                    )
                    y = grid_y * grid_size + random.randint(
                        -grid_size // 4, grid_size // 4
                    )
                elif attempts < 3 * max_attempts // 4:
                    # Strategy 3: Spiral outward from center or valid center
                    center_x, center_y = self._get_initial_center_position()
                    spiral_radius = (attempts - max_attempts // 2) * 2
                    angle = attempts * 0.1
                    x = int(center_x + spiral_radius * math.cos(angle))
                    y = int(center_y + spiral_radius * math.sin(angle))

                elif self.valid_positions is not None:
                    x, y = self._sample_valid_position(radius)
                else:
                    x = random.randint(radius, self.width - radius)
                    y = random.randint(radius, self.height - radius)

                # Check if position is valid (bounds + image boundaries)
                if not self._is_position_valid(x, y, radius):
                    attempts += 1
                    continue

                # Check for overlap with existing bubbles
                if not self._check_overlap(x, y, radius, positioned):
                    # Calculate distance to nearest edge (prefer center placement)
                    if self.valid_positions is not None:
                        # For image boundaries, prefer positions with more surrounding valid area
                        distance_to_edge = self._calculate_valid_area_score(
                            x, y, radius
                        )
                    else:
                        distance_to_edge = min(
                            x - radius,
                            y - radius,
                            self.width - x - radius,
                            self.height - y - radius,
                        )

                    if best_position is None or distance_to_edge > min_distance_to_edge:
                        best_position = (x, y)
                        min_distance_to_edge = distance_to_edge

                    # Accept good enough positions more readily
                    if distance_to_edge > radius // 6:
                        break

                attempts += 1

            # Place bubble if position found
            if best_position:
                x, y = best_position

                # Update color if using image colors
                final_color = color
                if use_image_colors and self.processed_image is not None:
                    final_color = self._sample_color_from_image(x, y, radius)

                positioned.append((word, count, word_type, final_color, radius, x, y))
            else:
                # This shouldn't happen often with the new algorithm
                failed.append((word, count, radius))

        # Report results
        success_rate = (
            len(positioned) / (len(positioned) + len(failed))
            if (len(positioned) + len(failed)) > 0
            else 0
        )
        print(
            f"Placed {len(positioned)} bubbles successfully ({success_rate:.1%} success rate)"
        )

        if failed:
            print(f"Failed to place {len(failed)} bubbles despite pre-selection")
            if len(failed) <= 5:
                failed_names = [f"{word}({count})" for word, count, _ in failed]
                print(f"Failed: {', '.join(failed_names)}")

        return positioned

    def _get_initial_center_position(self) -> Tuple[int, int]:
        """Get center position for first bubble, considering image boundaries."""
        if self.valid_positions is not None:
            # Use cached coordinates for MASSIVE performance improvement
            if self._valid_coords_cache is None or len(self._valid_coords_cache) == 0:
                # Fallback to center
                return self.width // 2, self.height // 2
            # Sample random valid position from cache - much faster than np.where every time
            x, y = random.choice(self._valid_coords_cache)
            return int(x), int(y)

        # Default to image center
        return self.width // 2, self.height // 2

    def _sample_valid_position(self, radius: int) -> Tuple[int, int]:
        """Sample a random position from valid areas - optimized for performance."""
        if self.valid_positions is None:
            return random.randint(radius, self.width - radius), random.randint(
                radius, self.height - radius
            )

        # Use cached coordinates for MASSIVE performance improvement
        if self._valid_coords_cache is None or len(self._valid_coords_cache) == 0:
            # Fallback to center
            return self.width // 2, self.height // 2

        # Sample random valid position from cache - much faster than np.where every time
        x, y = random.choice(self._valid_coords_cache)
        return int(x), int(y)

    def _calculate_valid_area_score(self, x: int, y: int, radius: int) -> float:
        """Calculate score based on amount of valid area around position."""
        if self.valid_positions is None:
            return min(
                x - radius,
                y - radius,
                self.width - x - radius,
                self.height - y - radius,
            )

        # Check area around the bubble
        check_radius = radius + 20  # Extra margin
        x1 = max(0, x - check_radius)
        y1 = max(0, y - check_radius)
        x2 = min(self.width, x + check_radius)
        y2 = min(self.height, y + check_radius)

        region = self.valid_positions[y1:y2, x1:x2]
        return np.sum(region) / max(
            1, region.size
        )  # Fraction of valid area around position

    def _check_overlap(
        self, x: int, y: int, radius: int, existing_bubbles: List[Tuple]
    ) -> bool:
        """Check if a bubble at (x, y) with given radius overlaps with existing bubbles."""
        for _, _, _, _, other_radius, other_x, other_y in existing_bubbles:
            distance = math.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)
            # Reduce padding for tighter packing, with minimum padding based on bubble size
            min_padding = max(2, min(radius, other_radius) // 10)
            if distance < radius + other_radius + min_padding:
                return True
        return False

    def _create_image(
        self,
        positioned_bubbles: List[Tuple],
        output_path: str,
        gradient_mode: bool = False,
    ) -> None:
        """Create and save the bubble chart image."""

        if gradient_mode:
            # Create gradient background
            gradient_background = self._create_gradient_background(positioned_bubbles)
            image = Image.fromarray(gradient_background)
            draw = ImageDraw.Draw(image)

            # Draw text labels over the gradient (no circle outlines)
            for word, count, word_type, color, radius, x, y in positioned_bubbles:
                # Draw text with sophisticated scaling based on actual text dimensions
                if radius > 8:  # Draw text for even smaller bubbles
                    # Use sophisticated text fitting algorithm
                    optimal_font_size, text_fits = self._find_optimal_font_size(
                        draw, word, radius
                    )

                    # Always try to show text, even if algorithm says it won't fit perfectly
                    if optimal_font_size >= 6:
                        if final_font := self._get_font(optimal_font_size):
                            try:
                                # Get actual text dimensions with final font
                                bbox = draw.textbbox((0, 0), word, font=final_font)
                                text_width = bbox[2] - bbox[0]
                                text_height = bbox[3] - bbox[1]

                                # Center text perfectly in bubble
                                text_x = x - text_width // 2
                                text_y = y - text_height // 2

                                # Add single subtle shadow for better readability on gradients
                                shadow_offset = max(1, optimal_font_size // 20)
                                if (
                                    optimal_font_size > 10
                                ):  # Add shadow for better contrast
                                    draw.text(
                                        (
                                            text_x + shadow_offset,
                                            text_y + shadow_offset,
                                        ),
                                        word,
                                        fill="white",
                                        font=final_font,
                                    )

                                # Main text
                                draw.text(
                                    (text_x, text_y),
                                    word,
                                    fill="black",
                                    font=final_font,
                                )
                            except Exception as e:
                                # Fallback - use simple positioning with guaranteed small font
                                fallback_size = max(6, min(12, radius // 4))
                                if fallback_font := self._get_font(fallback_size):
                                    draw.text(
                                        (
                                            x - len(word) * fallback_size // 4,
                                            y - fallback_size // 2,
                                        ),
                                        word,
                                        fill="black",
                                        font=fallback_font,
                                    )
                    elif minimal_font := self._get_font(6):
                        draw.text(
                            (x - len(word) * 2, y - 3),
                            word,
                            fill="black",
                            font=minimal_font,
                        )
        else:
            # Create base image (original mode)
            if (
                self.background_image_path
                and self.background_image is not None
                and self.show_background
            ):
                # Show background image when explicitly requested
                background_rgb = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(background_rgb)
            else:
                # Create image with white background (default for all other cases)
                image = Image.new("RGB", (self.width, self.height), "white")

            draw = ImageDraw.Draw(image)

            # Draw bubbles (original mode)
            for word, count, word_type, color, radius, x, y in positioned_bubbles:
                # Draw circle
                bbox = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(
                    bbox, fill=color, outline="black", width=max(1, radius // 50)
                )

                # Draw text with sophisticated scaling based on actual text dimensions
                if radius > 8:  # Draw text for even smaller bubbles
                    # Use sophisticated text fitting algorithm
                    optimal_font_size, text_fits = self._find_optimal_font_size(
                        draw, word, radius
                    )

                    # Always try to show text, even if algorithm says it won't fit perfectly
                    if optimal_font_size >= 6:
                        if final_font := self._get_font(optimal_font_size):
                            try:
                                # Get actual text dimensions with final font
                                bbox = draw.textbbox((0, 0), word, font=final_font)
                                text_width = bbox[2] - bbox[0]
                                text_height = bbox[3] - bbox[1]

                                # Center text perfectly in bubble
                                text_x = x - text_width // 2
                                text_y = y - text_height // 2

                                # Add subtle text shadow for better readability (original mode)
                                shadow_offset = max(1, optimal_font_size // 20)
                                if (
                                    optimal_font_size > 12
                                ):  # Only add shadow for larger text
                                    draw.text(
                                        (
                                            text_x + shadow_offset,
                                            text_y + shadow_offset,
                                        ),
                                        word,
                                        fill="white",
                                        font=final_font,
                                    )

                                # Main text
                                draw.text(
                                    (text_x, text_y),
                                    word,
                                    fill="black",
                                    font=final_font,
                                )
                            except Exception as e:
                                # Fallback - use simple positioning with guaranteed small font
                                fallback_size = max(6, min(12, radius // 4))
                                if fallback_font := self._get_font(fallback_size):
                                    draw.text(
                                        (
                                            x - len(word) * fallback_size // 4,
                                            y - fallback_size // 2,
                                        ),
                                        word,
                                        fill="black",
                                        font=fallback_font,
                                    )
                    elif minimal_font := self._get_font(6):
                        draw.text(
                            (x - len(word) * 2, y - 3),
                            word,
                            fill="black",
                            font=minimal_font,
                        )

        # Save image
        image.save(output_path, "PNG", quality=95, optimize=True)

    def _get_font(self, font_size: int):
        """Get a font of the specified size."""
        try:
            # Try to find a system font
            return ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            try:
                # Try other common fonts
                for font_name in ["calibri.ttf", "times.ttf", "georgia.ttf"]:
                    try:
                        return ImageFont.truetype(font_name, font_size)
                    except Exception:
                        continue
                # Fallback to default font
                return ImageFont.load_default()
            except Exception:
                return None

    def _find_optimal_font_size(self, draw, word: str, radius: int) -> Tuple[int, bool]:
        """
        Find the optimal font size that maximizes text size while fitting in the bubble.

        Returns:
            Tuple of (font_size, fits_successfully)
        """
        # Be more generous with usable area within the circle
        # Use more of the circle space for better text utilization
        max_text_width = radius * 1.8  # 90% of diameter - more generous
        max_text_height = radius * 1.2  # 60% of diameter - increased height allowance

        # More aggressive font size bounds for better utilization
        min_font_size = max(6, radius // 30)  # Scale more gradually with bubble
        max_font_size = min(500, radius * 0.75)  # Much more aggressive maximum

        # Ensure minimum bounds make sense
        if min_font_size > max_font_size:
            min_font_size = 6
            max_font_size = max(8, radius // 4)

        # Use binary search for efficient font size optimization
        best_font_size = min_font_size
        low, high = min_font_size, max_font_size

        while low <= high:
            mid_size = (low + high) // 2
            if font := self._get_font(mid_size):
                try:
                    # Measure actual text dimensions
                    bbox = draw.textbbox((0, 0), word, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Check if text fits - be more lenient
                    fits_width = text_width <= max_text_width
                    fits_height = text_height <= max_text_height

                    if fits_width and fits_height:
                        # Text fits, try larger font
                        best_font_size = mid_size
                        low = mid_size + 1
                    else:
                        # Text doesn't fit, try smaller font
                        high = mid_size - 1

                except Exception:
                    # Font measurement failed, try smaller
                    high = mid_size - 1
            else:
                # Font loading failed, try smaller
                high = mid_size - 1

        # Always return True for text fitting unless font size is impossibly small
        # This ensures bubbles always get text unless truly impossible
        return best_font_size, best_font_size >= 6

    def create_legend(
        self, output_path: str, exclude_types: Optional[List[str]] = None
    ) -> None:
        """
        Create a legend image showing word type colors.

        Args:
            output_path: Path to save the legend image
            exclude_types: List of word types to exclude from legend
        """
        # Filter out excluded word types from legend
        legend_colors = self.WORD_TYPE_COLORS.copy()
        if exclude_types:
            # Remove excluded types from legend
            for word_type in exclude_types:
                legend_colors.pop(word_type, None)

        # Remove unknown from legend display
        legend_colors.pop("unknown", None)

        if not legend_colors:
            print("No word types to display in legend (all excluded)")
            return

        legend_width = 400
        legend_height = (
            len(legend_colors) * 30 + 60
        )  # Extra space for title and exclusion info

        image = Image.new("RGB", (legend_width, legend_height), "white")
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()

        y_offset = 20
        draw.text((10, 5), "Word Type Legend", fill="black", font=title_font)

        # Add exclusion info if applicable
        if exclude_types:
            exclusion_text = f"Excluded types: {', '.join(exclude_types)}"
            draw.text((10, y_offset), exclusion_text, fill="gray", font=font)
            y_offset += 25

        for word_type, color in legend_colors.items():
            # Draw color circle
            draw.ellipse([10, y_offset, 30, y_offset + 20], fill=color, outline="black")

            # Draw label
            label = f"{word_type} - {self._get_word_type_name(word_type)}"
            draw.text((40, y_offset + 2), label, fill="black", font=font)

            y_offset += 30

        image.save(output_path, "PNG", quality=95)
        print(f"Legend saved to: {output_path}")

    def _get_word_type_name(self, word_type: str) -> str:
        """Get full name for word type abbreviation."""
        names = {
            "conj": "Conjunction",
            "art": "Article",
            "adj": "Adjective",
            "adv": "Adverb",
            "prep": "Preposition",
            "noun": "Noun",
            "verb": "Verb",
            "dpron": "Demonstrative Pronoun",
            "indpron": "Indefinite Pronoun",
            "intpron": "Interrogative Pronoun",
            "opron": "Other Pronoun",
            "ppron": "Personal Pronoun",
            "refpron": "Reflexive Pronoun",
            "relpron": "Relative Pronoun",
            "spron": "Subject Pronoun",
        }
        return names.get(word_type, word_type.capitalize())

    @staticmethod
    def get_available_word_types() -> List[str]:
        """Get available word types that can be excluded."""
        from config import DICTIONARY_CONFIG

        return DICTIONARY_CONFIG.get("word_types", [])

    def save_debug_images(self, output_dir: str) -> None:
        """Save debug images showing boundary mask and valid positions."""
        if not self.background_image_path:
            print("No background image to debug")
            return

        import os

        os.makedirs(output_dir, exist_ok=True)

        if self.boundary_mask is not None:
            # Save boundary mask
            boundary_img = (self.boundary_mask * 255).astype(np.uint8)
            boundary_pil = Image.fromarray(boundary_img, mode="L")
            boundary_path = os.path.join(output_dir, "boundary_mask.png")
            boundary_pil.save(boundary_path)
            print(f"Boundary mask saved to: {boundary_path}")

        if self.valid_positions is not None:
            # Save valid positions mask
            valid_img = (self.valid_positions * 255).astype(np.uint8)
            valid_pil = Image.fromarray(valid_img, mode="L")
            valid_path = os.path.join(output_dir, "valid_positions.png")
            valid_pil.save(valid_path)
            print(f"Valid positions mask saved to: {valid_path}")

        if self.processed_image is not None:
            # Save processed image
            processed_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            processed_pil = Image.fromarray(processed_rgb)
            processed_path = os.path.join(output_dir, "processed_image.png")
            processed_pil.save(processed_path)
            print(f"Processed image saved to: {processed_path}")

    @staticmethod
    def create_with_image_example(
        word_counts: Counter,
        dict_manager,
        background_image_path: str,
        output_path: str,
        use_image_colors: bool = True,
        exclude_types: Optional[List[str]] = None,
        width: int = None,
        height: int = None,
        debug_dir: str = None,
        use_boundaries: bool = True,
        show_background: bool = False,
        gradient_mode: bool = False,
    ) -> None:
        """
        Example method showing how to create bubble chart with image boundaries.

        Args:
            word_counts: Counter of word frequencies
            dict_manager: DictionaryManager for word type lookup
            background_image_path: Path to background image
            output_path: Path to save the final bubble chart
            use_image_colors: Whether to sample colors from the image
            exclude_types: Word types to exclude
            width: Canvas width (optional)
            height: Canvas height (optional)
            debug_dir: Directory to save debug images (optional)
            use_boundaries: Whether to use image boundaries for placement (optional)
            show_background: Whether to display background image (optional)
            gradient_mode: Whether to create gradient transitions between bubbles (optional)
        """
        boundary_mode = (
            "with image boundaries"
            if use_boundaries
            else "with image colors only (no boundaries)"
        )
        background_mode = " with visible background" if show_background else ""
        gradient_info = " with gradient transitions" if gradient_mode else ""
        print(
            f"Creating bubble chart {boundary_mode}{background_mode}{gradient_info}..."
        )

        # Create visualizer with background image
        visualizer = BubbleVisualizer(
            width=width,
            height=height,
            background_image_path=background_image_path,
            use_boundaries=use_boundaries,
            show_background=show_background,
        )

        # Save debug images if requested
        if debug_dir:
            visualizer.save_debug_images(debug_dir)

        # Create bubble chart
        visualizer.create_bubble_chart(
            word_counts=word_counts,
            dict_manager=dict_manager,
            output_path=output_path,
            exclude_types=exclude_types,
            use_image_colors=use_image_colors,
            gradient_mode=gradient_mode,
        )

        print("Image-based bubble chart creation complete!")

    def _resize_image_preserve_aspect(self, image):
        """Resize image to fit canvas while preserving aspect ratio."""
        img_height, img_width = image.shape[:2]

        # Calculate scale factor to fit within canvas
        scale_w = self.width / img_width
        scale_h = self.height / img_height
        scale = min(scale_w, scale_h)  # Use smaller scale to fit entirely

        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height))

        # Create canvas and center the image
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Calculate positioning to center the image
        start_y = (self.height - new_height) // 2
        start_x = (self.width - new_width) // 2

        # Place resized image on canvas
        canvas[start_y : start_y + new_height, start_x : start_x + new_width] = resized

        return canvas

    def _create_gradient_background(
        self, positioned_bubbles: List[Tuple]
    ) -> np.ndarray:
        """
        Create a gradient background by blending bubble colors with localized influence zones.
        Much faster and creates proper gradients between nearby bubbles only.

        Args:
            positioned_bubbles: List of positioned bubble data

        Returns:
            RGB image array with gradient background
        """
        print("Generating localized gradient background...")

        # Create background with neutral color
        background = np.full(
            (self.height, self.width, 3), 250, dtype=np.uint8
        )  # Light gray background

        if not positioned_bubbles:
            return background

        # Create influence map for each bubble separately (much faster)
        influence_map = np.zeros((self.height, self.width, 3), dtype=np.float32)
        total_weights = np.zeros((self.height, self.width), dtype=np.float32)

        print(f"Processing {len(positioned_bubbles)} bubbles for gradient...")

        for i, (word, count, word_type, color, radius, x, y) in enumerate(
            positioned_bubbles
        ):
            if i % 10 == 0:  # Progress indicator
                print(f"  Processing bubble {i+1}/{len(positioned_bubbles)}")

            # Convert hex color to RGB
            if color.startswith("#"):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            else:
                r, g, b = 128, 128, 128

            # Define influence radius (larger bubbles have more influence)
            influence_radius = int(
                radius * 2.5
            )  # Influence extends beyond bubble itself

            # Calculate bounding box for this bubble's influence
            x_min = max(0, x - influence_radius)
            x_max = min(self.width, x + influence_radius + 1)
            y_min = max(0, y - influence_radius)
            y_max = min(self.height, y + influence_radius + 1)

            # Create coordinate arrays for this region only
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max), np.arange(y_min, y_max), indexing="xy"
            )

            # Calculate distances from bubble center
            distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

            # Create smooth falloff weights
            # Strong influence within bubble radius, gradual falloff to influence_radius
            weights = np.zeros_like(distances)

            # Inside bubble: full strength
            inside_mask = distances <= radius
            weights[inside_mask] = 1.0

            # Outside bubble but within influence: smooth falloff
            outside_mask = (distances > radius) & (distances <= influence_radius)
            if np.any(outside_mask):
                falloff_distances = distances[outside_mask] - radius
                max_falloff = influence_radius - radius
                # Smooth cubic falloff
                normalized_falloff = falloff_distances / max_falloff
                weights[outside_mask] = (1 - normalized_falloff) ** 3

            # Apply weights to influence map
            region_weights = weights[..., np.newaxis]
            region_color = np.array([r, g, b], dtype=np.float32)

            influence_map[y_min:y_max, x_min:x_max] += region_weights * region_color
            total_weights[y_min:y_max, x_min:x_max] += weights

        # Normalize by total weights where we have influence
        mask = total_weights > 0.01  # Areas with significant influence

        # Apply gradient colors where we have influence
        for channel in range(3):
            background[mask, channel] = np.clip(
                influence_map[mask, channel] / total_weights[mask], 0, 255
            ).astype(np.uint8)

        # Areas with no influence keep the neutral background

        print("Localized gradient background generation complete")
        return background
