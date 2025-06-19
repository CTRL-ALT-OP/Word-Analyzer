"""
Main entry point for the Word Analyzer application.
Provides both command-line interface and example usage.
"""

import argparse
import json
import sys
from typing import List
import time

from word_analysis import WordAnalyzer
from dictionary_manager import DictionaryManager
from config import OUTPUT_CONFIG, DICTIONARY_CONFIG
from bubble_visualizer import BubbleVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Word Analyzer - Generate linguistic fingerprints from text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a PDF file
  python main.py --input document.pdf --author "Author Name"
  
  # Analyze a text file with custom pattern
  python main.py --input text.txt --pattern "noun verb adj" --author "Author Name"
  
  # Analyze specific pages of a PDF
  python main.py --input book.pdf --pdf-pages 10 100 --author "Author Name"
  
  # Create a bubble chart visualization
  python main.py --input text.txt --build-graph bubble_chart.png
  
  # Create bubble chart with legend
  python main.py --input text.txt --build-graph chart.png --graph-legend legend.png
  
  # Create bubble chart excluding articles and pronouns
  python main.py --input text.txt --build-graph chart.png --exclude-types art ppron dpron
  
  # Create image-based bubble chart (bubbles within portrait boundaries)
  python main.py --input text.txt --build-graph portrait_chart.png --background-image portrait.jpg
  
  # Create image-based bubble chart with colors sampled from image
  python main.py --input text.txt --build-graph colored_chart.png --background-image portrait.jpg --use-image-colors
  
  # Use image colors without boundary constraints (full canvas placement)
  python main.py --input text.txt --build-graph full_chart.png --background-image portrait.jpg --use-image-colors --no-boundaries
  
  # Use image colors with visible background (overlay effect)
  python main.py --input text.txt --build-graph overlay_chart.png --background-image photo.jpg --use-image-colors --no-boundaries --show-background
  
  # Create image-based chart with custom canvas size and debug images
  python main.py --input text.txt --build-graph chart.png --background-image photo.jpg --canvas-size 1920 1080 --debug-images debug/
  
  # List available word types for exclusion
  python main.py --list-types
  
  # Compare two fingerprints
  python main.py --compare fingerprint1.txt fingerprint2.txt
  
  # Clean dictionaries
  python main.py --clean-dictionaries
""",
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--input", "-i", type=str, help="Input file path or text string"
    )
    input_group.add_argument(
        "--input-type",
        "-t",
        type=str,
        choices=["pdf", "txt", "docx", "string"],
        help="Input type (auto-detected if not specified)",
    )
    input_group.add_argument(
        "--author", "-a", type=str, help="Author name for the analysis"
    )

    # Pattern options
    pattern_group = parser.add_argument_group("Pattern Options")
    pattern_group.add_argument(
        "--pattern",
        "-p",
        type=str,
        help="Fingerprint pattern (space-separated word types)",
    )
    pattern_group.add_argument(
        "--pattern-file", type=str, help="Load pattern from JSON file"
    )

    # PDF options
    pdf_group = parser.add_argument_group("PDF Options")
    pdf_group.add_argument(
        "--pdf-pages",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Page range for PDF extraction",
    )

    # Dictionary options
    dict_group = parser.add_argument_group("Dictionary Options")
    dict_group.add_argument("--dict-path", type=str, help="Path to dictionary files")
    dict_group.add_argument(
        "--clean-dictionaries",
        action="store_true",
        help="Remove duplicates from dictionaries",
    )
    dict_group.add_argument(
        "--dict-stats", action="store_true", help="Show dictionary statistics"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o", type=str, help="Save fingerprint to file"
    )
    output_group.add_argument(
        "--quiet", "-q", action="store_true", help="Minimal output"
    )
    output_group.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )
    output_group.add_argument(
        "--build-graph",
        type=str,
        help="Create bubble chart visualization and save to specified path",
    )
    output_group.add_argument(
        "--graph-legend",
        type=str,
        help="Save legend for bubble chart to specified path",
    )
    output_group.add_argument(
        "--exclude-types",
        nargs="+",
        type=str,
        help="Exclude word types from bubble chart (e.g., art conj prep)",
    )
    output_group.add_argument(
        "--list-types",
        action="store_true",
        help="List available word types for exclusion",
    )

    # Image-based bubble chart options
    image_group = parser.add_argument_group("Image-Based Bubble Chart Options")
    image_group.add_argument(
        "--background-image",
        type=str,
        help="Background image path for image-based bubble charts",
    )
    image_group.add_argument(
        "--use-image-colors",
        action="store_true",
        help="Sample colors from background image instead of using word type colors",
    )
    image_group.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Use image for colors only, ignore boundaries (allows full canvas placement)",
    )
    image_group.add_argument(
        "--show-background",
        action="store_true",
        help="Display background image in final chart (only works with --no-boundaries)",
    )
    image_group.add_argument(
        "--canvas-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Canvas size for bubble chart (default: 3840x2160 for 4K)",
    )
    image_group.add_argument(
        "--debug-images",
        type=str,
        help="Directory to save debug images showing boundary detection",
    )

    # Comparison options
    compare_group = parser.add_argument_group("Comparison Options")
    compare_group.add_argument(
        "--compare",
        nargs=2,
        type=str,
        metavar=("FILE1", "FILE2"),
        help="Compare two fingerprint files",
    )

    return parser.parse_args()


def load_pattern_from_file(filepath: str) -> List[str]:
    """Load pattern from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "pattern" in data:
            return data["pattern"]
        else:
            raise ValueError(
                "Pattern file must contain a list or dict with 'pattern' key"
            )


def main():
    """Main entry point."""
    args = parse_arguments()

    # Set quiet mode
    if args.quiet:
        OUTPUT_CONFIG["verbose"] = False
        OUTPUT_CONFIG["timing_info"] = False

    # Initialize dictionary manager
    dict_manager = DictionaryManager(dictionary_path=args.dict_path)

    # Handle list types option
    if args.list_types:
        word_types = BubbleVisualizer.get_available_word_types()
        print("\nAvailable word types for exclusion:")
        print("-" * 40)

        # Word type descriptions
        descriptions = {
            "conj": "Conjunction",
            "art": "Article",
            "adj": "Adjective",
            "adv": "Adverb",
            "prep": "Preposition",
            "noun": "Noun",
            "verb": "Verb",
            "dpron": "Demonstrative pronoun",
            "indpron": "Indefinite pronoun",
            "intpron": "Interrogative pronoun",
            "opron": "Other pronoun",
            "ppron": "Personal pronoun",
            "refpron": "Reflexive pronoun",
            "relpron": "Relative pronoun",
            "spron": "Subject pronoun",
        }

        for word_type in word_types:
            desc = descriptions.get(word_type, word_type.capitalize())
            print(f"{word_type:10s}: {desc}")
        print("-" * 40)
        print(f"Usage: --exclude-types {' '.join(word_types[:3])}")
        return

    # Handle dictionary operations
    if args.clean_dictionaries:
        print("Cleaning dictionaries...")
        dict_manager.load_dictionaries()
        removed = dict_manager.remove_duplicates()
        for word_type, count in removed.items():
            if count > 0:
                print(f"Removed {count} duplicates from {word_type}")
        dict_manager.save_dictionaries()
        return

    if args.dict_stats:
        dict_manager.load_dictionaries()
        stats = dict_manager.get_statistics()
        print("\nDictionary Statistics:")
        print("-" * 30)
        for word_type, count in sorted(stats.items()):
            print(f"{word_type:10s}: {count:6d} words")
        print("-" * 30)
        print(f"{'Total':10s}: {sum(stats.values()):6d} words")
        return

    # Handle comparison
    if args.compare:
        analyzer = WordAnalyzer(dict_manager)
        fp1 = analyzer.load_fingerprint(args.compare[0])
        fp2 = analyzer.load_fingerprint(args.compare[1])
        similarity = analyzer.compare_fingerprints(fp1, fp2)

        print(f"\nFingerprint Comparison:")
        print(f"File 1: {args.compare[0]}")
        print(f"File 2: {args.compare[1]}")
        print(f"Similarity: {similarity:.2%}")
        return

    # Regular analysis
    if not args.input:
        print("Error: --input is required for analysis")
        sys.exit(1)

    # Initialize analyzer
    analyzer = WordAnalyzer(dict_manager)

    # Prepare pattern
    pattern = None
    if args.pattern:
        pattern = args.pattern.split()
    elif args.pattern_file:
        pattern = load_pattern_from_file(args.pattern_file)

    # Prepare extraction kwargs
    extract_kwargs = {}
    if args.pdf_pages:
        extract_kwargs["page_range"] = tuple(args.pdf_pages)
    if args.input_type:
        extract_kwargs["input_type"] = args.input_type

    # Perform analysis

    start_time = time.time()

    if OUTPUT_CONFIG["verbose"] and args.author:
        print(f"Fingerprinting {args.author}...")

    # Analyze text
    word_counts = analyzer.analyze_text(args.input, **extract_kwargs)

    if OUTPUT_CONFIG["verbose"]:
        print(f"Analyzed {sum(word_counts.values())} words, {len(word_counts)} unique.")

    # Generate fingerprint
    fingerprint = analyzer.generate_fingerprint(pattern, word_counts)

    # Display results
    analyzer.display_fingerprint(fingerprint)

    if OUTPUT_CONFIG["timing_info"]:
        elapsed = round(time.time() - start_time, 2)
        print(f"\nCompleted fingerprint in {elapsed} seconds")

    # Save fingerprint
    if args.output:
        analyzer.save_fingerprint(args.output, fingerprint)
        print(f"\nFingerprint saved to: {args.output}")

    # Handle bubble chart generation
    if args.build_graph:
        try:
            # Check for image processing dependencies if background image is provided
            if hasattr(args, "background_image") and args.background_image:
                try:
                    import cv2
                    import numpy as np
                except ImportError:
                    print("Error: Image-based features require additional packages.")
                    print("Install with: pip install opencv-python numpy")
                    print(
                        "Or install all image requirements: pip install -r requirements_image.txt"
                    )
                    return

            # Determine canvas size
            width, height = None, None
            if hasattr(args, "canvas_size") and args.canvas_size:
                width, height = args.canvas_size

                # Create visualizer with optional background image
            background_image_path = getattr(args, "background_image", None)
            no_boundaries = getattr(args, "no_boundaries", False)
            show_background = getattr(args, "show_background", False)

            # Validate show_background usage
            if show_background and not no_boundaries:
                print(
                    "Warning: --show-background only works with --no-boundaries. Background will not be shown."
                )
                show_background = False

            visualizer = BubbleVisualizer(
                width=width,
                height=height,
                background_image_path=background_image_path,
                use_boundaries=(
                    background_image_path is not None and not no_boundaries
                ),
                show_background=show_background,
            )

            # Save debug images if requested
            if (
                hasattr(args, "debug_images")
                and args.debug_images
                and background_image_path
            ):
                import os

                os.makedirs(args.debug_images, exist_ok=True)
                visualizer.save_debug_images(args.debug_images)
                print(f"Debug images saved to: {args.debug_images}")

            # Prepare exclude types
            exclude_types = (
                args.exclude_types
                if hasattr(args, "exclude_types") and args.exclude_types
                else None
            )

            # Determine if using image colors
            use_image_colors = (
                hasattr(args, "use_image_colors")
                and args.use_image_colors
                and background_image_path is not None
            )

            # Create bubble chart
            print(f"Creating bubble chart with {len(word_counts)} unique words...")
            visualizer.create_bubble_chart(
                word_counts,
                dict_manager,
                args.build_graph,
                exclude_types=exclude_types,
                use_image_colors=use_image_colors,
            )

            # Create legend if specified
            if args.graph_legend:
                visualizer.create_legend(args.graph_legend, exclude_types=exclude_types)

                # Print summary of what was created
            if background_image_path:
                color_info = (
                    " with image colors"
                    if use_image_colors
                    else " with word type colors"
                )
                boundary_info = ""
                if no_boundaries:
                    if show_background:
                        boundary_info = " (full canvas, background visible)"
                    else:
                        boundary_info = " (full canvas, no boundaries)"
                else:
                    boundary_info = " (within image boundaries)"

                print(
                    f"SUCCESS: Created image-based bubble chart{color_info}{boundary_info}"
                )
                if hasattr(args, "debug_images") and args.debug_images:
                    print(f"DEBUG: Debug images available in: {args.debug_images}")
            else:
                print("SUCCESS: Created standard bubble chart")

        except ImportError as e:
            print(f"Error: {e}")
            if hasattr(args, "background_image") and args.background_image:
                print(
                    "Install required packages: pip install opencv-python numpy pillow matplotlib"
                )
            else:
                print("Install required packages: pip install Pillow matplotlib")
        except Exception as e:
            print(f"Error creating bubble chart: {e}")
            if hasattr(args, "background_image") and args.background_image:
                print(
                    "TIP: Check that the background image file exists and is a supported format (JPG, PNG, etc.)"
                )
                if hasattr(args, "debug_images") and args.debug_images:
                    print(
                        f"DEBUG: Check debug images in '{args.debug_images}' to troubleshoot boundary detection"
                    )

    # Handle output
    if args.json:
        output_data = {
            "fingerprint": [
                {"pattern": p, "word": w, "count": c} for p, w, c in fingerprint
            ],
            "sentence": " ".join(w for _, w, _ in fingerprint) + ".",
        }
        if args.author:
            output_data["author"] = args.author

        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
        else:
            print(json.dumps(output_data, indent=2))
    elif args.output:
        analyzer.save_fingerprint(args.output, fingerprint)
        print(f"\nFingerprint saved to: {args.output}")


def example_usage():
    """Example of programmatic usage."""
    # Create analyzer
    analyzer = WordAnalyzer()

    # Example 1: Analyze a text string
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    This sentence contains many different words.
    """

    print("Example 1: Analyzing sample text")
    word_counts = analyzer.analyze_text(sample_text, input_type="string")

    # Generate fingerprint with simple pattern
    simple_pattern = ["art", "adj", "noun", "verb"]
    fingerprint = analyzer.generate_fingerprint(simple_pattern, word_counts)
    analyzer.display_fingerprint(fingerprint)

    # Example 2: Use the default Edgar Allan Poe pattern
    print("\n\nExample 2: Using default pattern")
    default_fingerprint = analyzer.generate_fingerprint(word_counts=word_counts)
    analyzer.display_fingerprint(default_fingerprint)

    # Example 3: Custom pattern with length constraints
    print("\n\nExample 3: Custom pattern with length constraints")
    custom_pattern = [
        "shart",
        "lnoun",
        "verb",
        "madj",
    ]  # short article, long noun, verb, medium adjective
    custom_fingerprint = analyzer.generate_fingerprint(custom_pattern, word_counts)
    analyzer.display_fingerprint(custom_fingerprint)


if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Run example if no arguments provided
        print("No arguments provided. Running example usage...\n")
        example_usage()
        print("\n\nFor command-line usage, run: python main.py --help")
