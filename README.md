# Semester Named Entity Recognition (NER) System

A comprehensive AI system for analyzing semester information from PDF files using Tesseract OCR and Random Forest classification. This system can extract text from PDF documents containing images and identify different types of semester-related entities with high accuracy.

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Libraries Used](#libraries-used)
- [Configuration](#configuration)
- [Training the Model](#training-the-model)
- [Processing PDFs](#processing-pdfs)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)

## Features

- **PDF Text Extraction**: Uses Tesseract OCR to extract text from PDF files containing images
- **Named Entity Recognition**: Identifies and classifies semester-related entities using Random Forest
- **Multiple Semester Types**: Recognizes numeric semesters (1-8), seasonal semesters (Fall, Spring, Summer), and other academic terms
- **Batch Processing**: Process multiple PDF files simultaneously
- **High Accuracy**: Achieves >50% accuracy in identifying semester entities
- **Synthetic Data Generation**: Automatically generates training data when needed
- **Interactive Mode**: User-friendly command-line interface
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Model Persistence**: Save and load trained models
- **Evaluation Tools**: Built-in model evaluation and visualization

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux Ubuntu 18.04+
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space
- **Tesseract OCR**: Version 4.0 or higher

## Installation

### Step 1: Install Tesseract OCR

#### Windows:
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location: `C:\Program Files\Tesseract-OCR\`
3. Add Tesseract to your system PATH

#### macOS:
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

### Step 2: Install Python Dependencies

1. Clone or download this repository
2. Navigate to the project directory
3. Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv semester_ner_env

# Activate virtual environment
# Windows:
semester_ner_env\Scripts\activate
# macOS/Linux:
source semester_ner_env/bin/activate
```

4. Install required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Install Additional Dependencies

For PDF processing, you may need to install Poppler:

#### Windows:
1. Download Poppler from: https://poppler.freedesktop.org/
2. Extract and add to PATH

#### macOS:
```bash
brew install poppler
```

#### Linux:
```bash
sudo apt install poppler-utils
```

## Setup Instructions

### 1. Verify Installation

Test your installation by running:

```bash
python -c "import pytesseract, sklearn, cv2, nltk; print('All dependencies installed successfully!')"
```

### 2. Configure Tesseract Path

If Tesseract is not in your default location, update the path in `config.py`:

```python
TESSERACT_CMD = r'C:\Your\Path\To\tesseract.exe'  # Windows
# or
TESSERACT_CMD = '/usr/local/bin/tesseract'  # macOS/Linux
```

### 3. Download NLTK Data

The system will automatically download required NLTK data on first run, but you can do it manually:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
```

## Usage

### Quick Start

1. **Train the model** (uses synthetic data by default):
```bash
python train_model.py
```

2. **Process a single PDF**:
```bash
python main.py --mode predict --pdf-path "path/to/your/document.pdf"
```

3. **Run in interactive mode**:
```bash
python main.py
```

### Command Line Options

#### Training Mode
```bash
# Train with synthetic data only
python train_model.py

# Train with custom data
python train_model.py --train-data "path/to/training_data.json"

# Train without synthetic data augmentation
python train_model.py --no-synthetic --train-data "path/to/training_data.csv"

# Create sample training data
python train_model.py --create-sample-data
```

#### Processing Mode
```bash
# Process single PDF
python main.py --mode predict --pdf-path "document.pdf"

# Batch process multiple PDFs
python main.py --mode batch --pdf-dir "path/to/pdf/directory"

# Evaluate model performance
python main.py --mode evaluate --test-data "test_data.json"
```

### Interactive Mode

Run the application in interactive mode for a user-friendly experience:

```bash
python main.py
```

This will present you with options to:
1. Train a new model
2. Process single PDF files
3. Batch process multiple PDFs
4. Evaluate model performance
5. Check model status
6. Exit

## File Structure

```
semester-ner-system/
│
├── config.py                 # Configuration settings
├── main.py                   # Main application file
├── train_model.py            # Training script
├── pdf_processor.py          # PDF processing and OCR
├── text_preprocessor.py      # Text preprocessing and feature extraction
├── ner_model.py             # Random Forest NER model
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── data/                    # Training data directory
│   └── sample_training_data.json
│
├── models/                  # Saved models directory
│   ├── semester_ner_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── output/                  # Processing results
│   ├── *.json              # Individual PDF results
│   └── batch_processing_summary.json
│
└── temp/                   # Temporary files
```

## Libraries Used

### Core Libraries
- **pytesseract (0.3.10)**: OCR text extraction from images
- **scikit-learn (1.3.2)**: Random Forest classifier and ML utilities
- **opencv-python (4.8.1.78)**: Image preprocessing for better OCR
- **Pillow (10.0.1)**: Image handling and manipulation
- **nltk (3.8.1)**: Natural language processing and tokenization

### PDF Processing
- **PyPDF2 (3.0.1)**: PDF metadata extraction
- **pdf2image (1.16.3)**: Convert PDF pages to images

### Data Processing
- **pandas (2.1.4)**: Data manipulation and analysis
- **numpy (1.24.4)**: Numerical computing
- **joblib (1.3.2)**: Model serialization

### Visualization
- **matplotlib (3.8.2)**: Plotting and visualization
- **seaborn (0.13.0)**: Statistical data visualization

## Configuration

Key configuration options in `config.py`:

### OCR Settings
```python
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_CONFIG = '--oem 3 --psm 6'
PDF_DPI = 300
```

### Model Settings
```python
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
RANDOM_STATE = 42
```

### Feature Extraction
```python
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 3)
```

### Entity Labels
The system recognizes these semester entities:
- `SEMESTER_1` to `SEMESTER_8`: Numeric semesters
- `FALL_SEMESTER`, `SPRING_SEMESTER`, `SUMMER_SEMESTER`: Seasonal semesters
- `OTHER`: Non-semester entities

## Training the Model

### Option 1: Use Synthetic Data (Default)
```bash
python train_model.py
```

This generates 2000+ synthetic training samples automatically.

### Option 2: Use Custom Training Data

#### JSON Format:
```json
[
  {"text": "This course is offered in the first semester.", "label": "SEMESTER_1"},
  {"text": "Spring semester registration begins soon.", "label": "SPRING_SEMESTER"},
  {"text": "This is a general education course.", "label": "OTHER"}
]
```

#### CSV Format:
```csv
text,label
"This course is offered in the first semester.",SEMESTER_1
"Spring semester registration begins soon.",SPRING_SEMESTER
"This is a general education course.",OTHER
```

### Option 3: Use PDF Directory
```bash
python train_model.py --train-data "path/to/pdf/directory"
```

The system will extract text from all PDFs and generate labels automatically.

### Training with Custom Data:
```bash
python train_model.py --train-data "your_training_data.json"
```

## Processing PDFs

### Single PDF Processing
```bash
python main.py --mode predict --pdf-path "document.pdf"
```

### Batch Processing
```bash
python main.py --mode batch --pdf-dir "pdf_directory"
```

### Output Format
Results are saved as JSON files containing:
- Extracted text from all pages
- Identified sentences
- Semester entity predictions with confidence scores
- Statistics and metadata

Example output:
```json
{
  "pdf_info": {
    "file_name": "document.pdf",
    "page_count": 5,
    "total_sentences": 45
  },
  "semester_entities": [
    {
      "text": "This course is offered in the first semester.",
      "label": "SEMESTER_1",
      "confidence": 0.95,
      "entities": [...]
    }
  ],
  "statistics": {
    "total_semester_entities": 12,
    "avg_confidence": 0.87
  }
}
```

## API Reference

### Core Classes

#### `SemesterNERApplication`
Main application class that orchestrates all components.

**Methods:**
- `train_model(training_data_path, use_synthetic_data)`: Train the NER model
- `process_pdf(pdf_path, save_results)`: Process a single PDF file
- `batch_process_pdfs(pdf_directory, save_results)`: Process multiple PDFs
- `evaluate_model(test_data_path)`: Evaluate model performance

#### `PDFProcessor`
Handles PDF processing and OCR text extraction.

**Methods:**
- `extract_text_from_pdf(pdf_path)`: Extract text from PDF using OCR
- `batch_process_pdfs(pdf_directory)`: Process multiple PDFs
- `get_pdf_metadata(pdf_path)`: Get PDF metadata

#### `SemesterNERModel`
Random Forest-based NER model for semester entities.

**Methods:**
- `train(texts, labels, use_synthetic_data)`: Train the model
- `predict(texts)`: Make predictions on texts
- `evaluate(test_texts, test_labels)`: Evaluate model performance
- `save_model(model_path)`: Save trained model
- `load_model(model_path)`: Load trained model

#### `TextPreprocessor`
Text preprocessing and feature extraction.

**Methods:**
- `preprocess_text(text)`: Clean and preprocess text
- `extract_semester_entities(text)`: Extract entities using patterns
- `create_training_data(texts, labels)`: Create feature matrix
- `generate_synthetic_data(num_samples)`: Generate synthetic training data

## Troubleshooting

### Common Issues

#### 1. Tesseract Not Found
**Error**: `TesseractNotFoundError`

**Solution**:
- Ensure Tesseract is installed
- Update `TESSERACT_CMD` in `config.py` with correct path
- Add Tesseract to system PATH

#### 2. PDF Processing Fails
**Error**: `pdf2image.exceptions.PDFInfoNotInstalledError`

**Solution**:
- Install Poppler utilities
- Ensure poppler is in system PATH

#### 3. Memory Issues
**Error**: `MemoryError` during training

**Solution**:
- Reduce `MAX_FEATURES` in `config.py`
- Use smaller batch sizes
- Close other applications to free memory

#### 4. Low OCR Quality
**Problem**: Poor text extraction from PDFs

**Solution**:
- Increase `PDF_DPI` in `config.py` (try 400-600)
- Ensure PDF images are clear and high-resolution
- Check if PDFs contain actual images vs. text layers

#### 5. Import Errors
**Error**: `ModuleNotFoundError`

**Solution**:
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility

### Getting Help

1. Check the logs in the console for detailed error messages
2. Verify all dependencies are installed correctly
3. Test with sample data first
4. Check file permissions for input/output directories

## Performance

### Expected Performance Metrics
- **Accuracy**: >50% (typically 70-85% with synthetic data)
- **Processing Speed**: 1-2 pages per second (depends on image quality)
- **Memory Usage**: 500MB-2GB depending on PDF size and model complexity
- **Model Size**: ~50-100MB for trained model files

### Optimization Tips
1. **For better OCR**:
   - Use high-resolution PDFs (300+ DPI)
   - Ensure text is clear and not blurry
   - Consider image preprocessing parameters

2. **For better NER accuracy**:
   - Provide more training data
   - Use domain-specific training samples
   - Tune hyperparameters in `config.py`

3. **For faster processing**:
   - Reduce PDF DPI for faster (but less accurate) OCR
   - Use smaller Random Forest models
   - Process PDFs in parallel using batch mode

## Sample Commands

```bash
# Quick start - train and test
python train_model.py
python main.py --mode predict --pdf-path "sample.pdf"

# Train with custom data
python train_model.py --train-data "my_training_data.json" --output-dir "my_models"

# Process multiple PDFs
python main.py --mode batch --pdf-dir "documents" --output-dir "results"

# Evaluate model
python main.py --mode evaluate --test-data "test_set.csv"

# Interactive mode
python main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Tesseract OCR team for the excellent OCR engine
- scikit-learn contributors for machine learning tools
- NLTK team for natural language processing utilities

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional testing and validation with your specific data and requirements.