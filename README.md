# Word Analyzer - Linguistic Fingerprinting & Visualization Tool

A comprehensive tool for analyzing text patterns and creating stunning visualizations. Generate unique "linguistic fingerprints" from any text and visualize word patterns with advanced bubble charts.

## ✨ Features

- 🎯 **Two Interfaces**: Easy-to-use GUI and powerful command-line interface
- 📚 **Multiple Formats**: PDF, TXT, DOCX, and direct text input
- 🎨 **Advanced Visualizations**: Bubble charts with image boundary detection
- 🌈 **Color Sampling**: Extract colors from images for natural integration
- 🔧 **Pattern Builder**: Visual tool for creating custom analysis patterns
- 📊 **Dictionary Management**: Comprehensive word classification system

## 🚀 Quick Start

### Installation

```bash
# Install basic requirements
pip install -r requirements.txt

# For advanced image features (optional)
pip install opencv-python
```

### Option 1: GUI Interface (Recommended for Beginners)

```bash
# Launch the graphical interface
python test_gui.py
```

### Option 2: Command Line Interface

```bash
# Analyze a text file
python main.py --input document.pdf --author "Author Name"

# Create a bubble chart
python main.py --input text.txt --build-graph chart.png
```

## 🖥️ Using the GUI

### Getting Started with GUI

1. **Run the application**: `python test_gui.py`
2. **Load a text file**: Click "Load File" or try "Load Example"
3. **Switch between tabs**:
   - **Fingerprinting**: Create linguistic patterns
   - **Visualization**: Generate bubble charts

### Fingerprinting Tab

**Create Text Patterns:**
- Select word types from dropdowns (noun, verb, adjective, etc.)
- Add length constraints (short, medium, long words)
- Build patterns like: "article + adjective + noun + verb"
- See live results as you build patterns

**Example Pattern:**
- Word Type: `noun` | Length: `Long (≥5 chars)` | **Add**
- Result: Finds long nouns in your text

### Visualization Tab

**Create Stunning Charts:**
1. **Set canvas size**: Use 4K (3840×2160) for high quality or HD (1920×1080) for web
2. **Add background image** (optional): Upload a photo for boundary-based bubbles
3. **Choose options**:
   - ✅ Use image boundaries (bubbles follow image shape)
   - ✅ Sample colors from image (natural color palette)
   - ✅ Create legend file
4. **Generate preview**: Test your settings quickly
5. **Save visualization**: Export high-resolution PNG

## 💻 Command Line Usage

### Basic Analysis

```bash
# Analyze any text file
python main.py --input yourfile.txt --author "Author Name"

# Analyze specific PDF pages
python main.py --input book.pdf --pdf-pages 10 50 --author "Author"

# Use custom pattern
python main.py --input text.txt --pattern "art adj noun verb" --author "Author"
```

### Create Visualizations

```bash
# Basic bubble chart
python main.py --input text.txt --build-graph chart.png

# Image-based bubble chart
python main.py --input text.txt --build-graph chart.png --background-image photo.jpg

# With color sampling from image
python main.py --input text.txt --build-graph chart.png --background-image photo.jpg --use-image-colors

# Custom size with debug images
python main.py --input text.txt --build-graph chart.png --canvas-size 1920 1080 --debug-images debug/
```

### Utility Commands

```bash
# Show dictionary statistics
python main.py --dict-stats

# Clean dictionaries
python dictionary_cleaner.py --clean --save
```

## 🎨 Creating Great Visualizations

### Best Background Images

- **Portrait photos**: Work exceptionally well
- **High contrast**: Clear subject separation from background
- **Good lighting**: Well-lit subjects
- **Simple backgrounds**: Avoid cluttered scenes

### Word Type Filtering

For cleaner charts, exclude common words:
- `art` (articles): a, an, the
- `prep` (prepositions): in, on, at, by
- `conj` (conjunctions): and, but, or

### Pattern Examples

**Simple**: `art adj noun verb`
→ "The quick fox jumps"

**Complex**: `indpron prep adj conj adv verb`
→ "All of beautiful and more was" (literary style)

## 🔧 Pattern Syntax

### Word Types Available
- `noun` - Nouns (cat, house, idea)
- `verb` - Verbs (run, think, have)
- `adj` - Adjectives (big, red, beautiful)
- `adv` - Adverbs (quickly, very, well)
- `art` - Articles (a, an, the)
- `prep` - Prepositions (in, on, at, by)
- `conj` - Conjunctions (and, but, or)
- `ppron` - Personal pronouns (I, you, he, she)
- And many more...

### Length Constraints
Add prefixes to word types for size filtering:
- `sh` + word type = Short (≤2 characters): `shverb`
- `l` + word type = Long (≥5 characters): `lnoun`
- `ll` + word type = Very long (≥7 characters): `lladj`
- `m` + word type = Medium (≥4 characters): `mverb`
- `mm` + word type = Exactly 3 characters: `mmnoun`
- `ml` + word type = Exactly 4 characters: `mlverb`

**Example**: `shart lnoun verb` finds "short articles + long nouns + any verbs"

## 📁 File Structure

- `word_analyzer_gui.py` - Main GUI application
- `main.py` - Command-line interface
- `word_analyzer.py` - Core analysis engine
- `bubble_visualizer.py` - Visualization generator
- `test_gui.py` - GUI launcher script
- `Dict/` - Word classification dictionaries
- `texts/` - Example text files

## 🛠️ Troubleshooting

### GUI Won't Start
```bash
# Test your setup
python test_gui.py

# Install missing packages
pip install Pillow matplotlib
```

### Poor Image Boundary Detection
- Try different images with better contrast
- Use portrait photos instead of landscapes
- Enable debug images to see detection results
- Use "color sampling only" mode as alternative

### Slow Performance
- Use smaller canvas sizes for testing
- Exclude common word types
- Try preview mode before full generation

## 🚀 Advanced Usage

### Custom Dictionary Management

```python
from dictionary_manager import DictionaryManager

# Add new words to dictionaries
dict_manager = DictionaryManager()
dict_manager.load_dictionaries()
dict_manager.add_word("blockchain", "noun")
dict_manager.save_dictionaries()
```

### Compare Texts

```python
from word_analyzer import WordAnalyzer

analyzer = WordAnalyzer()

# Analyze two texts
fp1 = analyzer.generate_fingerprint(text1)
fp2 = analyzer.generate_fingerprint(text2)

# Compare similarity
similarity = analyzer.compare_fingerprints(fp1, fp2)
print(f"Similarity: {similarity:.2%}")
```

## 📖 Example Output

```
Fingerprinting Edgar Allan Poe...
Analyzed 267,453 words, 14,231 unique.

Fingerprint: All of these and more was never more.

                       Words: All | of | these | and  | more | was | never | more
Number of times word appears: 523 | 7821 | 187  | 5422 | 234  | 3901 | 432  | 234

Pattern Details:
 1. indpron         → All           (appears 523 times)
 2. shprep          → of            (appears 7821 times)
 3. ldpron          → these         (appears 187 times)
 4. conj            → and           (appears 5422 times)
 5. madv            → more          (appears 234 times)
 6. verb            → was           (appears 3901 times)
 7. ladv            → never         (appears 432 times)
 8. madv            → more          (appears 234 times)

Completed fingerprint in 3.24 seconds
```

## 📝 License

Open source tool for linguistic analysis and text visualization.
