"""
Bubble Chart Visualizer for Word Analysis
Creates bubble charts with words sized by frequency and colored by word type.
"""

from typing import List, Tuple, Counter, Optional

from PIL import Image, ImageDraw, ImageFont

PIL_AVAILABLE = False


class BubbleVisualizer:
    """Creates bubble chart visualizations of word frequency data."""

    def __init__(self, width: int = None, height: int = None):
        self.width = 100
        self.height = 100

    def create_bubble_chart(
        self,
        word_counts: Counter,
        dict_manager,
        output_path: str,
        min_bubble_size: int = 2,
        exclude_types: Optional[List[str]] = None,
    ) -> None:
        pass

    def _create_image(self, positioned_bubbles: List[Tuple], output_path: str) -> None:
        pass

    @staticmethod
    def get_available_word_types() -> List[str]:
        return []
