"""
Word Analyzer GUI
A comprehensive graphical interface for the word analysis tool with fingerprinting and visualization capabilities.
"""

import contextlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from tkinter.ttk import Notebook, Frame
import threading
from typing import Optional, Dict, List, Tuple, Counter
import os
from pathlib import Path
import json

# Import the existing modules
from word_analysis import WordAnalyzer
from dictionary_manager import DictionaryManager
from bubble_visualizer import BubbleVisualizer
from config import FINGERPRINT_CONFIG, DICTIONARY_CONFIG


class WordAnalyzerGUI:
    """Main GUI application for Word Analyzer."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Word Analyzer - Linguistic Fingerprinting & Visualization")
        self.root.geometry("1200x800")

        # Initialize components
        self.dict_manager = DictionaryManager()
        self.analyzer = WordAnalyzer(self.dict_manager)

        # Data storage
        self.current_file: Optional[str] = None
        self.word_counts: Optional[Counter] = None
        self.current_fingerprint: Optional[List[Tuple[str, str, int]]] = None
        self.current_pattern: List[str] = FINGERPRINT_CONFIG["default_pattern"].copy()

        # GUI state
        self.is_loading = False
        self.cached_preview_image = None  # Cache full-size preview for saving
        self.cached_preview_settings = None  # Cache settings used for preview

        # Create GUI
        self._create_menu()
        self._create_main_interface()

        # Load dictionaries in background
        self._load_dictionaries()

    def _create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Text File", command=self._load_file)
        file_menu.add_separator()
        file_menu.add_command(label="Save Fingerprint", command=self._save_fingerprint)
        file_menu.add_command(label="Load Fingerprint", command=self._load_fingerprint)
        file_menu.add_separator()
        file_menu.add_command(label="Save Pattern", command=self._save_pattern)
        file_menu.add_command(label="Load Pattern", command=self._load_pattern)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(
            label="Dictionary Statistics", command=self._show_dict_stats
        )
        tools_menu.add_command(
            label="Clean Dictionaries", command=self._clean_dictionaries
        )
        tools_menu.add_command(
            label="Compare Fingerprints", command=self._compare_fingerprints
        )
        tools_menu.add_separator()
        self.unknown_words_menu_item = tools_menu.add_command(
            label="View Unknown Words",
            command=self._show_unknown_words,
            state=tk.DISABLED,
        )
        # Store reference to tools menu for easier access
        self.tools_menu = tools_menu

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_interface(self):
        """Create the main interface with file selection and tabbed views."""
        # File selection frame
        file_frame = ttk.Frame(self.root)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="File:").pack(side=tk.LEFT)
        self.file_var = tk.StringVar(value="No file loaded")
        ttk.Label(file_frame, textvariable=self.file_var, foreground="blue").pack(
            side=tk.LEFT, padx=(5, 10)
        )

        ttk.Button(file_frame, text="Load File", command=self._load_file).pack(
            side=tk.LEFT
        )
        ttk.Button(file_frame, text="Load Example", command=self._load_example).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_bar = ttk.Progressbar(file_frame, mode="indeterminate")
        ttk.Label(file_frame, textvariable=self.progress_var).pack(side=tk.RIGHT)

        # Create notebook for tabs
        self.notebook = Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create tabs
        self._create_fingerprinting_tab()
        self._create_visualization_tab()

    def _create_fingerprinting_tab(self):
        """Create the fingerprinting tab."""
        self.fingerprint_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.fingerprint_frame, text="Fingerprinting")

        # Split into left and right panels
        main_paned = ttk.PanedWindow(self.fingerprint_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Pattern Builder (smaller weight)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Pattern Builder
        pattern_group = ttk.LabelFrame(left_frame, text="Pattern Builder", padding=10)
        pattern_group.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Available word types
        available_frame = ttk.Frame(pattern_group)
        available_frame.pack(fill=tk.X, pady=(0, 10))

        # Create scrollable frame for word types
        available_scroll = tk.Frame(available_frame)
        available_scroll.pack(fill=tk.X, pady=5)

        self.available_canvas = tk.Canvas(available_scroll, height=200)

        self.available_canvas.pack(side="left", fill="both", expand=True)

        # Add word type selection
        main_frame = self.available_canvas

        # Horizontal frame for dropdowns and button
        dropdown_frame = ttk.Frame(main_frame)
        dropdown_frame.pack()

        # Word type dropdown
        self.add_word_type_var = tk.StringVar()
        self.add_word_type_combo = ttk.Combobox(
            dropdown_frame,
            textvariable=self.add_word_type_var,
            values=DICTIONARY_CONFIG["word_types"],
            state="readonly",
            width=10,
        )
        self.add_word_type_combo.pack(side=tk.LEFT)

        # Length constraint dropdown
        self.add_length_var = tk.StringVar()
        length_options = [
            "No constraint",
            "Short (≤2 chars)",
            "Exactly 3 chars",
            "Exactly 4 chars",
            "Medium (≥4 chars)",
            "Long (≥5 chars)",
            "Very long (≥7 chars)",
        ]
        self.add_length_combo = ttk.Combobox(
            dropdown_frame,
            textvariable=self.add_length_var,
            values=length_options,
            state="readonly",
            width=20,
        )
        self.add_length_combo.pack(side=tk.LEFT)
        self.add_length_combo.current(0)  # Default to "No constraint"

        # Add button
        ttk.Button(dropdown_frame, text="Add", command=self._add_from_dropdowns).pack(
            side=tk.LEFT
        )

        # Separator
        ttk.Separator(main_frame, orient="horizontal").pack(fill=tk.X, pady=(10, 10))

        # Word type reference
        ttk.Label(
            main_frame, text="Word Type Reference:", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))

        # Word type descriptions with examples
        descriptions = {
            "conj": "Conjunction (and, but, or, so)",
            "art": "Article (a, an, the)",
            "adj": "Adjective (big, red, beautiful)",
            "adv": "Adverb (quickly, very, well)",
            "prep": "Preposition (in, on, at, by)",
            "noun": "Noun (cat, house, idea)",
            "verb": "Verb (run, think, have)",
            "dpron": "Demonstrative pronoun (this, that, these)",
            "indpron": "Indefinite pronoun (some, any, all)",
            "intpron": "Interrogative pronoun (who, what, which)",
            "opron": "Other pronoun (other pronouns)",
            "ppron": "Personal pronoun (I, you, he, she)",
            "refpron": "Reflexive pronoun (myself, yourself)",
            "relpron": "Relative pronoun (who, which, that)",
            "spron": "Subject pronoun (I, we, they)",
            "pnoun": "Proper noun (names, places)",
        }

        # Create reference frame with scrolling
        ref_frame = tk.Frame(main_frame)
        ref_frame.pack(fill=tk.BOTH, expand=True)

        ref_canvas = tk.Canvas(ref_frame, height=150)
        ref_scrollbar = ttk.Scrollbar(
            ref_frame, orient="vertical", command=ref_canvas.yview
        )
        ref_content = ttk.Frame(ref_canvas)

        ref_content.bind(
            "<Configure>",
            lambda e: ref_canvas.configure(scrollregion=ref_canvas.bbox("all")),
        )

        ref_canvas.create_window((0, 0), window=ref_content, anchor="nw")
        ref_canvas.configure(yscrollcommand=ref_scrollbar.set)

        ref_canvas.pack(side="left", fill="both", expand=True)
        ref_scrollbar.pack(side="right", fill="y")

        # Add descriptions
        for word_type in DICTIONARY_CONFIG["word_types"]:
            desc_text = f"{word_type}: {descriptions.get(word_type, word_type)}"
            ttk.Label(
                ref_content, text=desc_text, font=("Arial", 8), wraplength=250
            ).pack(anchor=tk.W, pady=1)

        # Current pattern
        pattern_frame = ttk.Frame(pattern_group)
        pattern_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Label(pattern_frame, text="Current Pattern:").pack(anchor=tk.W)

        self.pattern_canvas = tk.Canvas(
            pattern_frame, height=120, bg="white", relief=tk.SUNKEN, bd=1
        )
        pattern_scrollbar = ttk.Scrollbar(
            pattern_frame, orient="vertical", command=self.pattern_canvas.yview
        )
        self.pattern_scrollable_frame = ttk.Frame(self.pattern_canvas)

        self.pattern_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.pattern_canvas.configure(
                scrollregion=self.pattern_canvas.bbox("all")
            ),
        )

        self.pattern_canvas.create_window(
            (0, 0), window=self.pattern_scrollable_frame, anchor="nw"
        )
        self.pattern_canvas.configure(yscrollcommand=pattern_scrollbar.set)

        self.pattern_canvas.pack(side="left", fill="both", expand=True)
        pattern_scrollbar.pack(side="right", fill="y")

        # Pattern controls
        pattern_controls = ttk.Frame(pattern_group)
        pattern_controls.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            pattern_controls, text="Clear Pattern", command=self._clear_pattern
        ).pack(side=tk.LEFT)
        ttk.Button(
            pattern_controls, text="Reset to Default", command=self._reset_pattern
        ).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(
            pattern_controls,
            text="Generate Fingerprint",
            command=self._generate_fingerprint,
        ).pack(side=tk.RIGHT)

        # Right panel - Results (larger weight)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # Author input
        author_frame = ttk.Frame(right_frame)
        author_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(author_frame, text="Author:").pack(side=tk.LEFT)
        self.author_var = tk.StringVar()
        ttk.Entry(author_frame, textvariable=self.author_var, width=30).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Fingerprint display
        fp_group = ttk.LabelFrame(right_frame, text="Fingerprint Result", padding=10)
        fp_group.pack(fill=tk.BOTH, expand=True)

        # Fingerprint sentence
        sentence_frame = ttk.Frame(fp_group)
        sentence_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(sentence_frame, text="Sentence:").pack(anchor=tk.W)
        self.sentence_var = tk.StringVar(value="Generate a fingerprint to see results")
        sentence_label = ttk.Label(
            sentence_frame,
            textvariable=self.sentence_var,
            foreground="darkblue",
            font=("Arial", 12, "italic"),
        )
        sentence_label.pack(anchor=tk.W, pady=5)

        # Detailed results
        ttk.Label(fp_group, text="Detailed Results:").pack(anchor=tk.W)
        self.results_text = scrolledtext.ScrolledText(fp_group, height=15, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Update pattern display
        self._update_pattern_display()

    def _create_visualization_tab(self):
        """Create the visualization tab."""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")

        # Split into left controls and right preview
        viz_paned = ttk.PanedWindow(self.viz_frame, orient=tk.HORIZONTAL)
        viz_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Controls (much narrower)
        controls_frame = ttk.Frame(viz_paned)
        viz_paned.add(controls_frame, weight=1)

        # Canvas settings
        canvas_group = ttk.LabelFrame(
            controls_frame, text="Canvas Settings", padding=10
        )
        canvas_group.pack(fill=tk.X, pady=(0, 10))

        # Canvas size
        size_frame = ttk.Frame(canvas_group)
        size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(size_frame, text="Size:").pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value="3840")
        self.height_var = tk.StringVar(value="2160")
        ttk.Entry(size_frame, textvariable=self.width_var, width=8).pack(
            side=tk.LEFT, padx=(5, 2)
        )
        ttk.Label(size_frame, text="×").pack(side=tk.LEFT)
        ttk.Entry(size_frame, textvariable=self.height_var, width=8).pack(
            side=tk.LEFT, padx=(2, 5)
        )
        ttk.Button(
            size_frame, text="4K", command=lambda: self._set_canvas_size(3840, 2160)
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            size_frame, text="HD", command=lambda: self._set_canvas_size(1920, 1080)
        ).pack(side=tk.LEFT, padx=2)

        # Background image settings
        bg_group = ttk.LabelFrame(controls_frame, text="Background Image", padding=10)
        bg_group.pack(fill=tk.X, pady=(0, 10))

        # Background file selection
        bg_file_frame = ttk.Frame(bg_group)
        bg_file_frame.pack(fill=tk.X, pady=5)
        self.bg_file_var = tk.StringVar(value="No background image")
        ttk.Label(bg_file_frame, textvariable=self.bg_file_var, foreground="blue").pack(
            side=tk.LEFT
        )
        ttk.Button(
            bg_file_frame, text="Browse", command=self._load_background_image
        ).pack(side=tk.RIGHT)
        ttk.Button(
            bg_file_frame, text="Clear", command=self._clear_background_image
        ).pack(side=tk.RIGHT, padx=(0, 5))

        # Background options
        self.use_boundaries_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bg_group,
            text="Use image boundaries for bubble placement",
            variable=self.use_boundaries_var,
        ).pack(anchor=tk.W)

        self.use_image_colors_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bg_group,
            text="Sample colors from image",
            variable=self.use_image_colors_var,
        ).pack(anchor=tk.W)

        self.show_background_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bg_group,
            text="Show background image in final chart",
            variable=self.show_background_var,
        ).pack(anchor=tk.W)

        self.gradient_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bg_group,
            text="Gradient mode (smooth color transitions)",
            variable=self.gradient_mode_var,
        ).pack(anchor=tk.W)

        # Exclusion settings
        exclude_group = ttk.LabelFrame(
            controls_frame, text="Word Type Exclusions", padding=10
        )
        exclude_group.pack(fill=tk.X, pady=(0, 10))

        # Create checkboxes for word type exclusions
        self.exclude_vars = {}
        exclude_scroll_frame = tk.Frame(exclude_group)
        exclude_scroll_frame.pack(fill=tk.X)

        # Use a grid for better organization
        row = 0
        col = 0
        for word_type in DICTIONARY_CONFIG["word_types"]:
            var = tk.BooleanVar()
            self.exclude_vars[word_type] = var
            cb = ttk.Checkbutton(exclude_scroll_frame, text=word_type, variable=var)
            cb.grid(row=row, column=col, sticky="w", padx=5, pady=2)
            col += 1
            if col > 2:  # 3 columns
                col = 0
                row += 1

        # Common exclusions quick buttons
        exclude_buttons = ttk.Frame(exclude_group)
        exclude_buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            exclude_buttons, text="Exclude Common", command=self._exclude_common
        ).pack(side=tk.LEFT)
        ttk.Button(
            exclude_buttons, text="Clear Exclusions", command=self._clear_exclusions
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Output settings
        output_group = ttk.LabelFrame(
            controls_frame, text="Output Settings", padding=10
        )
        output_group.pack(fill=tk.X, pady=(0, 10))

        self.create_legend_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            output_group, text="Create legend file", variable=self.create_legend_var
        ).pack(anchor=tk.W)

        self.debug_images_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            output_group, text="Save debug images", variable=self.debug_images_var
        ).pack(anchor=tk.W)

        # Generation controls
        gen_group = ttk.LabelFrame(
            controls_frame, text="Generate Visualization", padding=10
        )
        gen_group.pack(fill=tk.X, pady=(0, 10))

        self.preview_var = tk.StringVar(value="No preview generated")
        ttk.Label(gen_group, textvariable=self.preview_var).pack(
            anchor=tk.W, pady=(0, 10)
        )

        gen_buttons = ttk.Frame(gen_group)
        gen_buttons.pack(fill=tk.X)
        ttk.Button(
            gen_buttons, text="Generate Preview", command=self._generate_preview
        ).pack(side=tk.LEFT)
        ttk.Button(
            gen_buttons, text="Save Visualization", command=self._save_visualization
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Right panel - Preview (much wider for 16:9 content)
        preview_frame = ttk.Frame(viz_paned)
        viz_paned.add(preview_frame, weight=3)

        self.preview_group = ttk.LabelFrame(preview_frame, text="Preview", padding=10)
        self.preview_group.pack(fill=tk.BOTH, expand=True)

        # Preview canvas - will be dynamically sized
        self.preview_canvas = tk.Canvas(
            self.preview_group, bg="white", relief=tk.SUNKEN, bd=1
        )
        self.preview_canvas.pack(expand=True)  # Center in container

        # Preview status
        self.preview_status_var = tk.StringVar(
            value="No preview available. Generate a visualization to see preview."
        )
        ttk.Label(
            self.preview_group, textvariable=self.preview_status_var, foreground="gray"
        ).pack(pady=10)

    def _add_from_dropdowns(self):
        """Add a word type to pattern from dropdown selections."""
        word_type = self.add_word_type_var.get()
        length_constraint = self.add_length_var.get()

        if not word_type:
            messagebox.showwarning("No Selection", "Please select a word type.")
            return

        # Map length constraint display to prefix
        length_map = {
            "No constraint": "",
            "Short (≤2 chars)": "sh",
            "Exactly 3 chars": "mm",
            "Exactly 4 chars": "ml",
            "Medium (≥4 chars)": "m",
            "Long (≥5 chars)": "l",
            "Very long (≥7 chars)": "ll",
        }

        prefix = length_map.get(length_constraint, "")
        pattern_element = prefix + word_type

        self._add_to_pattern(pattern_element)

        # Reset dropdowns for next addition
        self.add_word_type_var.set("")
        self.add_length_combo.current(0)

    def _create_tooltip(self, widget, text):
        """Create a simple tooltip for a widget."""

        def on_enter(event):
            # You could implement a proper tooltip here
            pass

        def on_leave(event):
            pass

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _update_pattern_display(self):
        """Update the visual display of the current pattern."""
        # Clear existing widgets
        for widget in self.pattern_scrollable_frame.winfo_children():
            widget.destroy()

        if not self.current_pattern:
            ttk.Label(
                self.pattern_scrollable_frame,
                text="Empty pattern - add word types from above",
                foreground="gray",
            ).pack(pady=20)
            return

        # Create draggable pattern elements
        for i, element in enumerate(self.current_pattern):
            element_frame = ttk.Frame(self.pattern_scrollable_frame)
            element_frame.pack(fill=tk.X, padx=5, pady=2)

            # Position label
            ttk.Label(element_frame, text=f"{i+1}:", width=3).pack(side=tk.LEFT)

            # Element button
            element_btn = ttk.Button(
                element_frame,
                text=element,
                command=lambda idx=i: self._edit_pattern_element(idx),
            )
            element_btn.pack(side=tk.LEFT, padx=5)

            # Move buttons
            if i > 0:
                ttk.Button(
                    element_frame,
                    text="↑",
                    width=3,
                    command=lambda idx=i: self._move_pattern_element(idx, -1),
                ).pack(side=tk.RIGHT)
            if i < len(self.current_pattern) - 1:
                ttk.Button(
                    element_frame,
                    text="↓",
                    width=3,
                    command=lambda idx=i: self._move_pattern_element(idx, 1),
                ).pack(side=tk.RIGHT)

            # Remove button
            ttk.Button(
                element_frame,
                text="×",
                width=3,
                command=lambda idx=i: self._remove_pattern_element(idx),
            ).pack(side=tk.RIGHT, padx=(5, 0))

    def _add_to_pattern(self, element: str):
        """Add an element to the current pattern."""
        self.current_pattern.append(element)
        self._update_pattern_display()
        self._auto_generate_fingerprint()

    def _remove_pattern_element(self, index: int):
        """Remove an element from the pattern."""
        if 0 <= index < len(self.current_pattern):
            self.current_pattern.pop(index)
            self._update_pattern_display()
            self._auto_generate_fingerprint()

    def _edit_pattern_element(self, index: int):
        """Edit a pattern element using dropdown menus."""
        if not 0 <= index < len(self.current_pattern):
            return
        current = self.current_pattern[index]

        # Parse current element to separate prefix and word type
        current_prefix = ""
        current_word_type = current

        # Check for length prefixes
        length_prefixes = ["sh", "mm", "ml", "m", "ll", "l"]
        for prefix in sorted(
            length_prefixes, key=len, reverse=True
        ):  # Check longer prefixes first
            if current.startswith(prefix):
                current_prefix = prefix
                current_word_type = current[len(prefix) :]
                break

        # Create edit dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Pattern Element")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Word type dropdown
        ttk.Label(dialog, text="Word Type:", font=("Arial", 10, "bold")).pack(
            pady=(20, 5)
        )
        word_type_var = tk.StringVar(value=current_word_type)
        word_type_combo = ttk.Combobox(
            dialog,
            textvariable=word_type_var,
            values=DICTIONARY_CONFIG["word_types"],
            state="readonly",
            width=30,
        )
        word_type_combo.pack(pady=5)

        # Length constraint dropdown
        ttk.Label(dialog, text="Length Constraint:", font=("Arial", 10, "bold")).pack(
            pady=(15, 5)
        )
        length_options = [
            ("No constraint", ""),
            ("Short (≤2 chars)", "sh"),
            ("Exactly 3 chars", "mm"),
            ("Exactly 4 chars", "ml"),
            ("Medium (≥4 chars)", "m"),
            ("Long (≥5 chars)", "l"),
            ("Very long (≥7 chars)", "ll"),
        ]

        length_var = tk.StringVar()
        length_combo = ttk.Combobox(
            dialog,
            textvariable=length_var,
            values=[opt[0] for opt in length_options],
            state="readonly",
            width=30,
        )
        length_combo.pack(pady=5)

        # Set initial length selection
        for i, (display, value) in enumerate(length_options):
            if value == current_prefix:
                length_combo.current(i)
                break

        result = {"cancelled": True}

        def on_ok():
            word_type = word_type_var.get()
            length_display = length_combo.get()

            # Find the actual prefix value
            length_prefix = ""
            for display, value in length_options:
                if display == length_display:
                    length_prefix = value
                    break

            new_element = length_prefix + word_type
            self.current_pattern[index] = new_element
            result["cancelled"] = False
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(
            side=tk.LEFT, padx=10
        )

        # Wait for dialog to close
        dialog.wait_window()

        if not result["cancelled"]:
            self._update_pattern_display()
            self._auto_generate_fingerprint()

    def _clear_pattern(self):
        """Clear the current pattern."""
        self.current_pattern = []
        self._update_pattern_display()
        self._auto_generate_fingerprint()

    def _reset_pattern(self):
        """Reset to default pattern."""
        self.current_pattern = FINGERPRINT_CONFIG["default_pattern"].copy()
        self._update_pattern_display()
        self._auto_generate_fingerprint()

    def _set_canvas_size(self, width: int, height: int):
        """Set canvas size to predefined values."""
        self.width_var.set(str(width))
        self.height_var.set(str(height))

    def _exclude_common(self):
        """Exclude commonly filtered word types."""
        common_exclusions = ["art", "prep", "conj"]
        for word_type in common_exclusions:
            if word_type in self.exclude_vars:
                self.exclude_vars[word_type].set(True)

    def _clear_exclusions(self):
        """Clear all word type exclusions."""
        for var in self.exclude_vars.values():
            var.set(False)

    def _load_dictionaries(self):
        """Load dictionaries in background."""

        def load():
            try:
                self.progress_var.set("Loading dictionaries...")
                self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
                self.progress_bar.start()

                self.dict_manager.load_dictionaries()

                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.progress_var.set("Ready")
            except Exception as e:
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.progress_var.set("Error loading dictionaries")
                messagebox.showerror("Error", f"Failed to load dictionaries: {e}")

        threading.Thread(target=load, daemon=True).start()

    def _load_file(self):
        """Load a text file for analysis."""
        if file_path := filedialog.askopenfilename(
            title="Select text file",
            filetypes=[
                ("All supported", "*.txt *.pdf *.docx"),
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("All files", "*.*"),
            ],
        ):
            self._load_and_analyze_file(file_path)

    def _load_example(self):
        """Load the example placeholder file."""
        example_path = "texts/placeholder.txt"
        if os.path.exists(example_path):
            self._load_and_analyze_file(example_path)
        else:
            messagebox.showwarning(
                "Example Not Found",
                "The example file 'texts/placeholder.txt' was not found.",
            )

    def _load_and_analyze_file(self, file_path: str):
        """Load and analyze a file in the background."""

        def analyze():
            try:
                self.is_loading = True
                self.progress_var.set("Analyzing file...")
                self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
                self.progress_bar.start()

                # Analyze the file
                self.word_counts = self.analyzer.analyze_text(file_path)
                self.current_file = file_path

                # Update UI
                self.root.after(0, lambda: self._on_file_loaded(file_path))

            except Exception as e:
                self.root.after(0, lambda: self._on_file_error(str(e)))
            finally:
                self.is_loading = False
                self.root.after(0, self._loading_finished)

        threading.Thread(target=analyze, daemon=True).start()

    def _on_file_loaded(self, file_path: str):
        """Called when file loading is complete."""
        filename = os.path.basename(file_path)
        word_count = sum(self.word_counts.values()) if self.word_counts else 0
        unique_count = len(self.word_counts) if self.word_counts else 0

        self.file_var.set(f"{filename} ({word_count:,} words, {unique_count:,} unique)")

        # Enable unknown words menu item
        try:
            self.tools_menu.entryconfig("View Unknown Words", state=tk.NORMAL)
        except Exception as e:
            print(f"Could not enable unknown words menu: {e}")

        # Auto-generate fingerprint if pattern exists
        self._auto_generate_fingerprint()

        messagebox.showinfo(
            "File Loaded",
            f"Successfully analyzed {filename}\n{word_count:,} total words\n{unique_count:,} unique words",
        )

    def _on_file_error(self, error_msg: str):
        """Called when file loading fails."""
        self.file_var.set("Failed to load file")
        messagebox.showerror("Error", f"Failed to load file: {error_msg}")

    def _loading_finished(self):
        """Clean up loading UI."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_var.set("Ready")

    def _generate_fingerprint(self):
        """Generate a fingerprint from the current pattern."""
        if not self.word_counts:
            messagebox.showwarning("No Data", "Please load a text file first.")
            return

        if not self.current_pattern:
            messagebox.showwarning("No Pattern", "Please create a pattern first.")
            return

        try:
            # Generate fingerprint
            self.current_fingerprint = self.analyzer.generate_fingerprint(
                pattern=self.current_pattern, word_counts=self.word_counts
            )

            # Update display
            words = [item[1] for item in self.current_fingerprint]
            sentence = " ".join(words) + "."
            self.sentence_var.set(sentence)

            # Update detailed results
            self.results_text.delete(1.0, tk.END)

            if author := self.author_var.get().strip():
                self.results_text.insert(tk.END, f"Fingerprinting {author}...\n\n")

            self.results_text.insert(tk.END, f"Fingerprint: {sentence}\n\n")

            # Add detailed table
            if self.current_fingerprint:
                max_len = max(len(item[1]) for item in self.current_fingerprint)

                # Headers
                self.results_text.insert(tk.END, "                       Words: ")
                self.results_text.insert(
                    tk.END,
                    " | ".join(
                        f"{item[1]:<{max_len}}" for item in self.current_fingerprint
                    ),
                )
                self.results_text.insert(tk.END, "\n")

                self.results_text.insert(tk.END, "Number of times word appears: ")
                self.results_text.insert(
                    tk.END,
                    " | ".join(
                        f"{item[2]:<{max_len}}" for item in self.current_fingerprint
                    ),
                )
                self.results_text.insert(tk.END, "\n\n")

                # Pattern details
                self.results_text.insert(tk.END, "Pattern Details:\n")
                for i, (pattern, word, count) in enumerate(self.current_fingerprint):
                    self.results_text.insert(
                        tk.END,
                        f"{i+1:2d}. {pattern:15s} → {word:15s} (appears {count} times)\n",
                    )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate fingerprint: {e}")

    def _load_background_image(self):
        """Load a background image for visualization."""
        if file_path := filedialog.askopenfilename(
            title="Select background image",
            filetypes=[
                (
                    "Image files",
                    "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.webp *.jfif",
                ),
                ("JPEG files", "*.jpg *.jpeg *.jfif"),
                ("PNG files", "*.png"),
                ("TIFF files", "*.tiff *.tif"),
                ("WebP files", "*.webp"),
                ("All files", "*.*"),
            ],
        ):
            self.bg_file_var.set(os.path.basename(file_path))
            self._background_image_path = file_path
            # Clear cached preview when background changes
            self.cached_preview_image = None
            self.cached_preview_settings = None

    def _clear_background_image(self):
        """Clear the background image."""
        self.bg_file_var.set("No background image")
        self._background_image_path = None
        # Clear cached preview when background changes
        self.cached_preview_image = None
        self.cached_preview_settings = None

    def _generate_preview(self):
        """Generate a preview of the visualization."""
        if not self.word_counts:
            messagebox.showwarning("No Data", "Please load a text file first.")
            return

        # Show progress indication
        self.preview_status_var.set("Generating preview...")
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(
            self.preview_canvas.winfo_width() / 2,
            self.preview_canvas.winfo_height() / 2,
            text="Generating preview...\nPlease wait...",
            font=("Arial", 14, "bold"),
            fill="blue",
            anchor=tk.CENTER,
        )
        self.root.update()

        def generate():
            try:
                # Get current settings
                current_settings = {
                    "width": int(self.width_var.get()),
                    "height": int(self.height_var.get()),
                    "background_image": getattr(self, "_background_image_path", None),
                    "use_boundaries": self.use_boundaries_var.get(),
                    "use_image_colors": self.use_image_colors_var.get(),
                    "show_background": self.show_background_var.get(),
                    "gradient_mode": self.gradient_mode_var.get(),
                    "excluded_types": [
                        word_type
                        for word_type, var in self.exclude_vars.items()
                        if var.get()
                    ],
                    "word_counts_hash": hash(
                        frozenset(self.word_counts.items())
                        if self.word_counts
                        else frozenset()
                    ),
                }

                # Check if we can use cached preview
                if (
                    self.cached_preview_image is not None
                    and self.cached_preview_settings == current_settings
                    and os.path.exists(self.cached_preview_image)
                ):
                    self.root.after(0, lambda: self._show_preview_from_cache())
                    return

                # Generate full-size visualization
                background_image = current_settings["background_image"]
                use_boundaries = current_settings["use_boundaries"] and background_image
                use_image_colors = (
                    current_settings["use_image_colors"] and background_image
                )
                show_background = current_settings["show_background"]
                gradient_mode = current_settings["gradient_mode"]
                excluded_types = current_settings["excluded_types"]

                # Create full-size visualizer
                visualizer = BubbleVisualizer(
                    width=current_settings["width"],
                    height=current_settings["height"],
                    background_image_path=background_image,
                    use_boundaries=use_boundaries,
                    show_background=show_background,
                )

                # Generate full-size image
                preview_path = "temp_preview_full.png"
                visualizer.create_bubble_chart(
                    word_counts=self.word_counts,
                    dict_manager=self.dict_manager,
                    output_path=preview_path,
                    exclude_types=excluded_types or None,
                    use_image_colors=use_image_colors,
                    gradient_mode=gradient_mode,
                )

                # Cache the full-size image
                self.cached_preview_image = preview_path
                self.cached_preview_settings = current_settings

                # Update UI
                self.root.after(0, lambda: self._show_preview(preview_path))

            except Exception as e:
                self.root.after(0, lambda: self._preview_error(str(e)))

        threading.Thread(target=generate, daemon=True).start()

    def _show_preview(self, preview_path: str):
        """Show the generated preview (scaled from full-size image) in a 16:9 optimized layout."""
        try:
            from PIL import Image, ImageTk

            # Load full-size image
            img = Image.open(preview_path)
            original_size = img.size

            # Get available space in preview container
            self.preview_group.update_idletasks()
            container_width = self.preview_group.winfo_width()
            container_height = self.preview_group.winfo_height()

            # Calculate optimal display size (maintaining aspect ratio)
            # Aim for 16:9 preview area with some padding
            max_width = max(400, container_width - 40)  # Leave padding
            max_height = max(225, container_height - 60)  # Leave padding for status

            # Maintain image aspect ratio
            img_ratio = original_size[0] / original_size[1]
            container_ratio = max_width / max_height

            if img_ratio > container_ratio:
                # Image is wider than container ratio
                display_width = max_width
                display_height = int(max_width / img_ratio)
            else:
                # Image is taller than container ratio
                display_height = max_height
                display_width = int(max_height * img_ratio)

            # Resize image
            img_copy = img.copy()
            img_copy = img_copy.resize(
                (display_width, display_height), Image.Resampling.LANCZOS
            )

            # Convert to tkinter format
            self.preview_photo = ImageTk.PhotoImage(img_copy)

            # Resize canvas to fit image exactly (no scrollbars needed)
            self.preview_canvas.configure(width=display_width, height=display_height)

            # Clear canvas and add image centered
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                display_width // 2,
                display_height // 2,
                anchor=tk.CENTER,
                image=self.preview_photo,
            )

            self.preview_status_var.set(
                f"Preview: {display_width}×{display_height} (full size: {original_size[0]}×{original_size[1]})"
            )

        except Exception as e:
            self.preview_status_var.set(f"Error displaying preview: {e}")

    def _show_preview_from_cache(self):
        """Show preview from cached full-size image."""
        if self.cached_preview_image and os.path.exists(self.cached_preview_image):
            self._show_preview(self.cached_preview_image)
        else:
            self.preview_status_var.set("Cached preview not available")

    def _save_visualization(self):
        """Save the final visualization."""
        if not self.word_counts:
            messagebox.showwarning("No Data", "Please load a text file first.")
            return

        # Check if we have a cached preview that matches current settings
        current_settings = {
            "width": int(self.width_var.get()),
            "height": int(self.height_var.get()),
            "background_image": getattr(self, "_background_image_path", None),
            "use_boundaries": self.use_boundaries_var.get(),
            "use_image_colors": self.use_image_colors_var.get(),
            "show_background": self.show_background_var.get(),
            "gradient_mode": self.gradient_mode_var.get(),
            "excluded_types": [
                word_type for word_type, var in self.exclude_vars.items() if var.get()
            ],
            "word_counts_hash": hash(
                frozenset(self.word_counts.items()) if self.word_counts else frozenset()
            ),
        }

        # Get output file
        output_path = filedialog.asksaveasfilename(
            title="Save visualization",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )

        if not output_path:
            return

        # If we have a cached preview with matching settings, use it
        if (
            self.cached_preview_image is not None
            and self.cached_preview_settings == current_settings
            and os.path.exists(self.cached_preview_image)
        ):

            with contextlib.suppress(Exception):
                # Copy cached image to output location
                import shutil

                shutil.copy2(self.cached_preview_image, output_path)

                # Create legend if requested
                if self.create_legend_var.get():
                    # Generate legend separately since it's quick
                    excluded_types = current_settings["excluded_types"]
                    background_image = current_settings["background_image"]
                    use_boundaries = (
                        current_settings["use_boundaries"] and background_image
                    )
                    show_background = current_settings["show_background"]

                    visualizer = BubbleVisualizer(
                        width=current_settings["width"],
                        height=current_settings["height"],
                        background_image_path=background_image,
                        use_boundaries=use_boundaries,
                        show_background=show_background,
                    )

                    legend_path = f"{os.path.splitext(output_path)[0]}_legend.png"
                    visualizer.create_legend(
                        legend_path,
                        exclude_types=excluded_types or None,
                    )

                self._visualization_complete(output_path)
                return

        # Generate new visualization
        self.progress_var.set("Generating visualization...")
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
        self.progress_bar.start()

        def generate():
            try:
                background_image = current_settings["background_image"]
                use_boundaries = current_settings["use_boundaries"] and background_image
                use_image_colors = (
                    current_settings["use_image_colors"] and background_image
                )
                show_background = current_settings["show_background"]
                gradient_mode = current_settings["gradient_mode"]
                excluded_types = current_settings["excluded_types"]

                # Create visualizer
                visualizer = BubbleVisualizer(
                    width=current_settings["width"],
                    height=current_settings["height"],
                    background_image_path=background_image,
                    use_boundaries=use_boundaries,
                    show_background=show_background,
                )

                # Save debug images if requested
                if self.debug_images_var.get() and background_image:
                    debug_dir = f"{os.path.splitext(output_path)[0]}_debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    visualizer.save_debug_images(debug_dir)

                # Generate visualization
                visualizer.create_bubble_chart(
                    word_counts=self.word_counts,
                    dict_manager=self.dict_manager,
                    output_path=output_path,
                    exclude_types=excluded_types or None,
                    use_image_colors=use_image_colors,
                    gradient_mode=gradient_mode,
                )

                # Update cache
                self.cached_preview_image = "temp_preview_full.png"
                self.cached_preview_settings = current_settings
                if os.path.exists(output_path):
                    import shutil

                    shutil.copy2(output_path, self.cached_preview_image)

                # Create legend if requested
                if self.create_legend_var.get():
                    legend_path = f"{os.path.splitext(output_path)[0]}_legend.png"
                    visualizer.create_legend(
                        legend_path,
                        exclude_types=excluded_types or None,
                    )

                self.root.after(0, lambda: self._visualization_complete(output_path))

            except Exception as e:
                self.root.after(0, lambda: self._visualization_error(str(e)))
            finally:
                self.root.after(0, self._loading_finished)

        threading.Thread(target=generate, daemon=True).start()

    def _visualization_complete(self, output_path: str):
        """Called when visualization generation is complete."""
        legend_info = " and legend" if self.create_legend_var.get() else ""
        debug_info = " (debug images saved)" if self.debug_images_var.get() else ""
        messagebox.showinfo(
            "Success",
            f"Visualization{legend_info} saved to:\n{output_path}{debug_info}",
        )

    def _visualization_error(self, error_msg: str):
        """Handle visualization generation errors."""
        messagebox.showerror("Error", f"Failed to generate visualization: {error_msg}")

    # Menu action methods
    def _save_fingerprint(self):
        """Save the current fingerprint."""
        if not self.current_fingerprint:
            messagebox.showwarning("No Fingerprint", "Generate a fingerprint first.")
            return

        if file_path := filedialog.asksaveasfilename(
            title="Save fingerprint",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        ):
            try:
                self.analyzer.save_fingerprint(file_path, self.current_fingerprint)
                messagebox.showinfo("Success", f"Fingerprint saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save fingerprint: {e}")

    def _load_fingerprint(self):
        """Load a fingerprint from file."""
        if file_path := filedialog.askopenfilename(
            title="Load fingerprint",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        ):
            try:
                fingerprint = self.analyzer.load_fingerprint(file_path)
                self.current_fingerprint = fingerprint

                # Update display
                words = [item[1] for item in fingerprint]
                sentence = " ".join(words) + "."
                self.sentence_var.set(sentence)

                # Update results
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Loaded fingerprint: {sentence}\n\n")

                for i, (pattern, word, count) in enumerate(fingerprint):
                    self.results_text.insert(
                        tk.END,
                        f"{i+1:2d}. {pattern:15s} → {word:15s} (count: {count})\n",
                    )

                messagebox.showinfo("Success", f"Fingerprint loaded from: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load fingerprint: {e}")

    def _save_pattern(self):
        """Save the current pattern."""
        if not self.current_pattern:
            messagebox.showwarning("No Pattern", "Create a pattern first.")
            return

        if file_path := filedialog.asksaveasfilename(
            title="Save pattern",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        ):
            try:
                with open(file_path, "w") as f:
                    json.dump({"pattern": self.current_pattern}, f, indent=2)
                messagebox.showinfo("Success", f"Pattern saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save pattern: {e}")

    def _load_pattern(self):
        """Load a pattern from file."""
        if file_path := filedialog.askopenfilename(
            title="Load pattern",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        ):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.current_pattern = data
                    elif isinstance(data, dict) and "pattern" in data:
                        self.current_pattern = data["pattern"]
                    else:
                        raise ValueError("Invalid pattern file format")

                self._update_pattern_display()
                messagebox.showinfo("Success", f"Pattern loaded from: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load pattern: {e}")

    def _show_dict_stats(self):
        """Show dictionary statistics."""
        try:
            stats = self.dict_manager.get_statistics()

            stats_window = tk.Toplevel(self.root)
            stats_window.title("Dictionary Statistics")
            stats_window.geometry("400x500")

            text_widget = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            text_widget.insert(tk.END, "Dictionary Statistics\n")
            text_widget.insert(tk.END, "=" * 30 + "\n\n")

            for word_type, count in sorted(stats.items()):
                text_widget.insert(tk.END, f"{word_type:15s}: {count:6,d} words\n")

            text_widget.insert(tk.END, "\n" + "-" * 30 + "\n")
            text_widget.insert(
                tk.END, f"{'Total':15s}: {sum(stats.values()):6,d} words\n"
            )

            text_widget.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to get dictionary statistics: {e}")

    def _clean_dictionaries(self):
        """Clean dictionaries (remove duplicates)."""
        if messagebox.askyesno(
            "Clean Dictionaries",
            "This will remove duplicate entries from the dictionaries. Continue?",
        ):
            try:
                removed = self.dict_manager.remove_duplicates()
                self.dict_manager.save_dictionaries()

                result_text = "Dictionary cleaning complete:\n\n"
                for word_type, count in removed.items():
                    if count > 0:
                        result_text += f"Removed {count} duplicates from {word_type}\n"

                if not any(removed.values()):
                    result_text += "No duplicates found."

                messagebox.showinfo("Success", result_text)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clean dictionaries: {e}")

    def _compare_fingerprints(self):
        """Compare two fingerprints."""
        if not self.current_fingerprint:
            messagebox.showwarning(
                "No Fingerprint", "Generate or load a fingerprint first."
            )
            return

        if file_path := filedialog.askopenfilename(
            title="Select fingerprint to compare with",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        ):
            try:
                other_fingerprint = self.analyzer.load_fingerprint(file_path)
                similarity = self.analyzer.compare_fingerprints(
                    self.current_fingerprint, other_fingerprint
                )

                result_window = tk.Toplevel(self.root)
                result_window.title("Fingerprint Comparison")
                result_window.geometry("600x400")

                text_widget = scrolledtext.ScrolledText(result_window, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                text_widget.insert(tk.END, "Fingerprint Comparison Results\n")
                text_widget.insert(tk.END, "=" * 40 + "\n\n")

                text_widget.insert(
                    tk.END,
                    f"Current fingerprint: {' '.join(item[1] for item in self.current_fingerprint)}.\n",
                )
                text_widget.insert(
                    tk.END,
                    f"Loaded fingerprint: {' '.join(item[1] for item in other_fingerprint)}.\n\n",
                )

                text_widget.insert(tk.END, f"Similarity: {similarity:.2%}\n")

                text_widget.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to compare fingerprints: {e}")

    def _show_about(self):
        """Show about dialog."""
        about_text = """Word Analyzer GUI v1.0

A comprehensive tool for linguistic fingerprinting and word visualization.

Features:
• Linguistic fingerprint generation
• Advanced bubble chart visualizations
• Image-based boundary detection
• Color sampling from images
• Pattern creation and editing
• File format support: PDF, TXT, DOCX

Created for advanced text analysis and visualization.
"""
        messagebox.showinfo("About Word Analyzer", about_text)

    def _auto_generate_fingerprint(self):
        """Automatically generate fingerprint when pattern changes."""
        if self.word_counts and self.current_pattern:
            with contextlib.suppress(Exception):
                self._generate_fingerprint()

    def _show_unknown_words(self):
        """Show unknown words that aren't in any dictionary."""
        if not self.word_counts:
            messagebox.showwarning("No Data", "Please load a text file first.")
            return

        # Find unknown words
        unknown_words = []
        for word in self.word_counts.keys():
            if self.dict_manager.get_word_type(word) is None:
                unknown_words.append((word, self.word_counts[word]))

        # Sort by frequency (most common first)
        unknown_words.sort(key=lambda x: x[1], reverse=True)

        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Unknown Words")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Summary label
        summary_text = f"Found {len(unknown_words)} unknown words out of {len(self.word_counts)} total unique words"
        ttk.Label(dialog, text=summary_text, font=("Arial", 12, "bold")).pack(pady=10)

        # Create frame for word list and assignment
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Word list with scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview for word list
        columns = ("Word", "Count", "Type")
        self.unknown_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=15
        )

        # Configure column headings and widths
        self.unknown_tree.heading("Word", text="Word")
        self.unknown_tree.heading("Count", text="Frequency")
        self.unknown_tree.heading("Type", text="Assigned Type")

        self.unknown_tree.column("Word", width=200, anchor="w")
        self.unknown_tree.column("Count", width=100, anchor="center")
        self.unknown_tree.column("Type", width=150, anchor="center")

        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.unknown_tree.yview
        )
        self.unknown_tree.configure(yscrollcommand=scrollbar.set)

        self.unknown_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Populate the list
        for word, count in unknown_words:
            self.unknown_tree.insert("", "end", values=(word, count, "Unknown"))

        # Assignment controls
        assign_frame = ttk.Frame(main_frame)
        assign_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(assign_frame, text="Assign selected word to category:").pack(
            side=tk.LEFT
        )

        # Word type dropdown
        self.assign_var = tk.StringVar()
        assign_combo = ttk.Combobox(
            assign_frame,
            textvariable=self.assign_var,
            values=DICTIONARY_CONFIG["word_types"],
            state="readonly",
            width=15,
        )
        assign_combo.pack(side=tk.LEFT, padx=(5, 0))

        def assign_word():
            selection = self.unknown_tree.selection()
            if not selection or not self.assign_var.get():
                messagebox.showwarning(
                    "No Selection", "Please select a word and choose a word type."
                )
                return

            item = selection[0]
            word = self.unknown_tree.item(item, "values")[0]
            word_type = self.assign_var.get()

            # Add word to dictionary
            try:
                self.dict_manager.add_word(word, word_type)
                # Save dictionaries to files
                self.dict_manager.save_dictionaries()
                # Update display
                self.unknown_tree.item(
                    item,
                    values=(word, self.unknown_tree.item(item, "values")[1], word_type),
                )
                messagebox.showinfo(
                    "Success",
                    f"Added '{word}' as {word_type} and saved to dictionary files",
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add word: {e}")

        ttk.Button(assign_frame, text="Assign", command=assign_word).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = WordAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
