"""
Configuration file for Named Entity Recognition (NER) system
Contains all necessary settings, paths, and parameters
"""

import os

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Tesseract OCR settings
TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Default Windows path
TESSERACT_CONFIG = '--oem 3 --psm 6'  # OCR Engine Mode and Page Segmentation Mode

# PDF processing settings
PDF_DPI = 300  # DPI for converting PDF pages to images
IMAGE_FORMAT = 'PNG'

# Text preprocessing settings
MIN_WORD_LENGTH = 2
MAX_SENTENCE_LENGTH = 1000
STOPWORDS_LANGUAGE = 'english'

# Random Forest settings
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
RANDOM_STATE = 42

# Feature extraction settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 3)  # Unigrams, bigrams, and trigrams

# Semester entity labels
SEMESTER_LABELS = [
    'SEMESTER_1', 'SEMESTER_2', 'SEMESTER_3', 'SEMESTER_4',
    'SEMESTER_5', 'SEMESTER_6', 'SEMESTER_7', 'SEMESTER_8',
    'FALL_SEMESTER', 'SPRING_SEMESTER', 'SUMMER_SEMESTER',
    'OTHER'  # For non-semester entities
]

# Model settings
MODEL_NAME = 'semester_ner_model.pkl'
VECTORIZER_NAME = 'tfidf_vectorizer.pkl'
LABEL_ENCODER_NAME = 'label_encoder.pkl'

# Training settings
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
CROSS_VALIDATION_FOLDS = 5

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' 