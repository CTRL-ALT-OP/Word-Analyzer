"""
Bubble Chart Visualizer for Word Analysis
Creates bubble charts with words sized by frequency and colored by word type.
"""

import math
import random
from typing import Dict, List, Tuple, Counter, Optional
from collections import defaultdict

try:
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.colors as mcolors

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from config import DICTIONARY_CONFIG


class BubbleVisualizer:
    """Creates bubble chart visualizations of word frequency data."""

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
        "unknown": "#BDC3C7",  # Gray - Unknown type
    }

    def __init__(self, width: int = None, height: int = None):
        """Initialize the bubble visualizer."""
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL (Pillow) is required for bubble visualization. Install with: pip install Pillow matplotlib"
            )

        self.width = width or self.DEFAULT_WIDTH
        self.height = height or self.DEFAULT_HEIGHT
        self.bubbles = []  # List of (x, y, radius, word, color) tuples

    def create_bubble_chart(
        self,
        word_counts: Counter,
        dict_manager,
        output_path: str,
        min_bubble_size: int = 2,
        exclude_types: Optional[List[str]] = None,
    ) -> None:
        """
        Create a bubble chart from word frequency data.

        Args:
            word_counts: Counter of word frequencies
            dict_manager: DictionaryManager for word type lookup
            output_path: Path to save the image
            min_bubble_size: Minimum bubble size in pixels
            exclude_types: List of word types to exclude (e.g., ["art", "conj", "prep"])
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

        # Calculate bubble sizes
        bubble_data = self._calculate_bubble_sizes(word_data)

        # Position bubbles (avoid overlap)
        positioned_bubbles = self._position_bubbles(bubble_data)

        # Create and save image
        self._create_image(positioned_bubbles, output_path)

        excluded_info = ""
        if exclude_types:
            excluded_info = f" (excluded types: {', '.join(exclude_types)})"

        print(f"Bubble chart saved to: {output_path}")
        print(
            f"Generated {len(positioned_bubbles)} bubbles from {len(word_counts)} words{excluded_info}"
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
        for i in range(20):  # More granular steps
            if max_freq > min_freq:
                # Logarithmic spacing to cover the full range
                log_min = math.log(min_freq) if min_freq > 0 else 0
                log_max = math.log(max_freq)
                log_cutoff = log_min + (log_max - log_min) * (1 - i / 19)
                cutoff = max(1, int(math.exp(log_cutoff)))
                test_cutoffs.append(cutoff)

        # Strategy 3: Linear intervals in different ranges
        test_cutoffs.extend(iter(range(1, min(21, max_freq + 1))))
        # Remove duplicates and sort
        test_cutoffs = sorted(set(test_cutoffs), reverse=True)

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

            # Calculate a score that balances word count and success rate
            # Prefer higher word counts, but require reasonable success rate
            if success_rate >= 0.75:  # Minimum viable success rate
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
                        f"  New best - Cutoff {test_cutoff}: {len(test_words)} words → {placeable_count} placeable ({success_rate:.1%}, score: {score:.1f})"
                    )
            elif len(test_cutoffs) <= 20:  # Show details for smaller searches
                print(
                    f"  Cutoff {test_cutoff}: {len(test_words)} words → {placeable_count} placeable ({success_rate:.1%}) - below threshold"
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

                if expanded_rate >= 0.75:  # More aggressive - accept 75% success rate
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

                            if subset_rate >= 0.75:
                                final_words = test_subset
                                print(
                                    f"  Post-processing: added frequency {freq_level}+ words → {len(final_words)} total ({subset_rate:.1%} success)"
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

        # Test placement with faster algorithm
        return self._test_placement_fast(bubble_data)

    def _calculate_bubble_sizes_fast(
        self, word_data: List[Tuple[str, int, str, str]]
    ) -> List[Tuple[str, int, int]]:
        """Fast bubble size calculation for testing."""
        if not word_data:
            return []

        max_count = max(count for _, count, _, _ in word_data)
        min_count = min(count for _, count, _, _ in word_data)

        min_radius = 15
        max_radius = min(400, min(self.width, self.height) // 6)

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
        max_attempts = 100  # Much faster testing

        for word, count, radius in sorted_bubbles:
            # Find a position using grid
            found_position = False

            for attempt in range(max_attempts):
                if attempt < max_attempts // 2:
                    # Try systematic grid positions first
                    grid_x = (
                        attempt * 7
                    ) % grid_width  # Prime number for good distribution
                    grid_y = (attempt * 11) % grid_height
                else:
                    # Then try random
                    grid_x = random.randint(0, grid_width - 1)
                    grid_y = random.randint(0, grid_height - 1)

                # Convert to pixel coordinates
                pixel_x = grid_x * grid_size + grid_size // 2
                pixel_y = grid_y * grid_size + grid_size // 2

                # Check bounds
                if (
                    pixel_x - radius < 0
                    or pixel_x + radius >= self.width
                    or pixel_y - radius < 0
                    or pixel_y + radius >= self.height
                ):
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
                    break

            if not found_position:
                # If we can't place this bubble, we probably can't place smaller ones either
                # in this density, so break early
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
        """Calculate bubble radii based on word frequency using area-based scaling."""
        if not word_data:
            return []

        max_count = max(count for _, count, _, _ in word_data)
        min_count = min(count for _, count, _, _ in word_data)

        # Calculate total area available (use 80% to leave some margin)
        total_area = 0.8 * self.width * self.height
        total_frequency = sum(count for _, count, _, _ in word_data)

        # Scale radii much more aggressively - from 15 to 400 pixels
        min_radius = 15
        max_radius = min(
            400, min(self.width, self.height) // 6
        )  # Cap at 1/6 of smallest dimension

        bubble_data = []
        for word, count, word_type, color in word_data:
            if max_count == min_count:
                radius = max_radius // 2
            else:
                # Use square root scaling for area-proportional bubbles
                # This makes the visual representation more proportional to frequency
                area_ratio = count / total_frequency
                target_area = (
                    area_ratio * total_area * 0.3
                )  # Use 30% of available area for scaling

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
        self, bubble_data: List[Tuple[str, int, str, str, int]]
    ) -> List[Tuple[str, int, str, str, int, int, int]]:
        """Position bubbles with high success rate since we pre-selected an optimal set."""
        if not bubble_data:
            return []

        # Sort by size (largest first)
        sorted_bubbles = sorted(bubble_data, key=lambda x: x[4], reverse=True)
        positioned = []
        failed = []

        max_attempts = 5000  # More attempts since we expect high success

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
                        # First bubble goes in center
                        x = self.width // 2
                        y = self.height // 2

                elif attempts < max_attempts // 2:
                    # Strategy 2: Systematic grid coverage
                    grid_size = 30
                    grid_x = (attempts * 7) % (
                        self.width // grid_size
                    )  # Prime for distribution
                    grid_y = (attempts * 11) % (self.height // grid_size)
                    x = grid_x * grid_size + random.randint(
                        -grid_size // 4, grid_size // 4
                    )
                    y = grid_y * grid_size + random.randint(
                        -grid_size // 4, grid_size // 4
                    )

                elif attempts < 3 * max_attempts // 4:
                    # Strategy 3: Spiral outward from center
                    spiral_radius = (attempts - max_attempts // 2) * 3
                    angle = attempts * 0.1
                    x = int(self.width // 2 + spiral_radius * math.cos(angle))
                    y = int(self.height // 2 + spiral_radius * math.sin(angle))

                else:
                    # Strategy 4: Random placement as last resort
                    x = random.randint(radius, self.width - radius)
                    y = random.randint(radius, self.height - radius)

                # Ensure position is within bounds
                x = max(radius, min(x, self.width - radius))
                y = max(radius, min(y, self.height - radius))

                # Check for overlap with existing bubbles
                if not self._check_overlap(x, y, radius, positioned):
                    # Calculate distance to nearest edge (prefer center placement)
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

            # Since we pre-selected optimal set, we should succeed more often
            if best_position:
                x, y = best_position
                positioned.append((word, count, word_type, color, radius, x, y))
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

    def _create_image(self, positioned_bubbles: List[Tuple], output_path: str) -> None:
        """Create and save the bubble chart image."""
        # Create image with white background
        image = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(image)

        # Draw bubbles
        for word, count, word_type, color, radius, x, y in positioned_bubbles:
            # Draw circle
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, fill=color, outline="black", width=max(1, radius // 50))

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

                            # Add subtle text shadow for better readability
                            shadow_offset = max(1, optimal_font_size // 20)
                            if (
                                optimal_font_size > 12
                            ):  # Only add shadow for larger text
                                draw.text(
                                    (text_x + shadow_offset, text_y + shadow_offset),
                                    word,
                                    fill="white",
                                    font=final_font,
                                )
                            draw.text(
                                (text_x, text_y), word, fill="black", font=final_font
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
                    except:
                        continue
                # Fallback to default font
                return ImageFont.load_default()
            except:
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
