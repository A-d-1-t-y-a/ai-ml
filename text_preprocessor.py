"""
Text Preprocessing Module
Handles text cleaning, tokenization, and feature extraction for NER
"""

import re
import string
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {str(e)}")


class TextPreprocessor:
    """
    Handles text cleaning, preprocessing, and feature extraction for NER
    """
    
    def __init__(self):
        """
        Initialize text preprocessor with required components
        """
        try:
            # Initialize NLTK components
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words(config.STOPWORDS_LANGUAGE))
            
            # Initialize vectorizer and label encoder
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=config.MAX_FEATURES,
                ngram_range=config.NGRAM_RANGE,
                stop_words='english',
                lowercase=True,
                strip_accents='ascii'
            )
            
            self.label_encoder = LabelEncoder()
            
            # Semester-related keywords for pattern matching
            self.semester_patterns = {
                'numeric_semester': r'\b(?:semester|sem)\s*[1-8](?:st|nd|rd|th)?\b',
                'ordinal_semester': r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\s*semester\b',
                'season_semester': r'\b(?:fall|spring|summer|winter)\s*(?:semester|term)\b',
                'year_semester': r'\b(?:freshman|sophomore|junior|senior)\s*(?:year|semester)\b',
                'academic_year': r'\b(?:20\d{2}[-/]20\d{2})\s*(?:semester|academic year)\b'
            }
            
            logger.info("Text Preprocessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Text Preprocessor: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess raw text
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep alphanumeric, spaces, and basic punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        try:
            if not text:
                return []
            
            # Use NLTK sentence tokenizer
            sentences = sent_tokenize(text)
            
            # Filter out very short sentences
            filtered_sentences = [
                sentence for sentence in sentences 
                if len(sentence.split()) >= 3
            ]
            
            return filtered_sentences
            
        except Exception as e:
            logger.error(f"Error extracting sentences: {str(e)}")
            return [text] if text else []
    
    def tokenize_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            List[str]: List of tokens
        """
        try:
            if not text:
                return []
            
            # Tokenize using NLTK
            tokens = word_tokenize(text)
            
            # Convert to lowercase and filter
            tokens = [token.lower() for token in tokens if token.isalnum()]
            
            # Remove stopwords if requested
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Filter by minimum length
            tokens = [token for token in tokens if len(token) >= config.MIN_WORD_LENGTH]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {str(e)}")
            return []
    
    def extract_semester_entities(self, text: str) -> List[Dict[str, any]]:
        """
        Extract semester-related entities from text using pattern matching
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict]: List of detected semester entities
        """
        entities = []
        
        try:
            if not text:
                return entities
            
            # Convert to lowercase for pattern matching
            text_lower = text.lower()
            
            # Check each semester pattern
            for pattern_name, pattern in self.semester_patterns.items():
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in matches:
                    entity = {
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'pattern_type': pattern_name,
                        'label': self._classify_semester_entity(match.group())
                    }
                    entities.append(entity)
            
            # Remove duplicates and overlapping entities
            entities = self._remove_overlapping_entities(entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting semester entities: {str(e)}")
            return []
    
    def _classify_semester_entity(self, entity_text: str) -> str:
        """
        Classify semester entity into appropriate label
        
        Args:
            entity_text (str): Entity text
            
        Returns:
            str: Classification label
        """
        entity_lower = entity_text.lower()
        
        # Check for specific semester numbers
        for i in range(1, 9):
            if f'semester {i}' in entity_lower or f'sem {i}' in entity_lower:
                return f'SEMESTER_{i}'
        
        # Check for season-based semesters
        if 'fall' in entity_lower:
            return 'FALL_SEMESTER'
        elif 'spring' in entity_lower:
            return 'SPRING_SEMESTER'
        elif 'summer' in entity_lower:
            return 'SUMMER_SEMESTER'
        
        # Check for ordinal semesters
        ordinal_map = {
            'first': 'SEMESTER_1', 'second': 'SEMESTER_2', 'third': 'SEMESTER_3',
            'fourth': 'SEMESTER_4', 'fifth': 'SEMESTER_5', 'sixth': 'SEMESTER_6',
            'seventh': 'SEMESTER_7', 'eighth': 'SEMESTER_8'
        }
        
        for ordinal, label in ordinal_map.items():
            if ordinal in entity_lower:
                return label
        
        return 'OTHER'
    
    def _remove_overlapping_entities(self, entities: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Remove overlapping entities, keeping the longest match
        
        Args:
            entities (List[Dict]): List of entities
            
        Returns:
            List[Dict]: Filtered entities without overlaps
        """
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x['start'])
        
        # Remove overlaps
        filtered_entities = []
        for entity in entities:
            # Check if this entity overlaps with any previously added entity
            overlap = False
            for existing_entity in filtered_entities:
                if (entity['start'] < existing_entity['end'] and 
                    entity['end'] > existing_entity['start']):
                    # There's an overlap - keep the longer entity
                    if (entity['end'] - entity['start']) > (existing_entity['end'] - existing_entity['start']):
                        filtered_entities.remove(existing_entity)
                    else:
                        overlap = True
                    break
            
            if not overlap:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def create_training_data(self, texts: List[str], labels: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data from texts and labels
        
        Args:
            texts (List[str]): List of input texts
            labels (List[str]): List of corresponding labels (optional)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix and label array
        """
        try:
            if not texts:
                return np.array([]), np.array([])
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Extract features using TF-IDF
            X = self.tfidf_vectorizer.fit_transform(processed_texts)
            
            # Encode labels if provided
            if labels:
                y = self.label_encoder.fit_transform(labels)
            else:
                # Generate labels using pattern matching
                y = []
                for text in processed_texts:
                    entities = self.extract_semester_entities(text)
                    if entities:
                        y.append(entities[0]['label'])  # Use first detected entity
                    else:
                        y.append('OTHER')
                y = self.label_encoder.fit_transform(y)
            
            return X.toarray(), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating training data: {str(e)}")
            return np.array([]), np.array([])
    
    def transform_text(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            np.ndarray: Feature matrix
        """
        try:
            if not texts:
                return np.array([])
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Transform using fitted vectorizer
            X = self.tfidf_vectorizer.transform(processed_texts)
            
            return X.toarray()
            
        except Exception as e:
            logger.error(f"Error transforming text: {str(e)}")
            return np.array([])
    
    def extract_features(self, text: str) -> Dict[str, any]:
        """
        Extract various features from text for NER
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Dictionary of extracted features
        """
        try:
            if not text:
                return {}
            
            # Basic text features
            features = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(self.extract_sentences(text)),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'punctuation_count': sum(1 for char in text if char in string.punctuation),
                'digit_count': sum(1 for char in text if char.isdigit()),
                'uppercase_count': sum(1 for char in text if char.isupper())
            }
            
            # Semester-specific features
            semester_entities = self.extract_semester_entities(text)
            features.update({
                'semester_entity_count': len(semester_entities),
                'has_semester_keywords': any(
                    keyword in text.lower() 
                    for keyword in ['semester', 'term', 'academic', 'year', 'course']
                ),
                'has_numeric_semester': bool(re.search(r'\b(?:semester|sem)\s*[1-8]\b', text.lower())),
                'has_season_semester': bool(re.search(r'\b(?:fall|spring|summer|winter)\b', text.lower()))
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[List[str], List[str]]:
        """
        Generate synthetic training data for semester NER
        
        Args:
            num_samples (int): Number of samples to generate
            
        Returns:
            Tuple[List[str], List[str]]: Generated texts and labels
        """
        try:
            texts = []
            labels = []
            
            # Template patterns for generating synthetic data
            semester_templates = [
                "This course is offered in {semester}.",
                "Students must complete {semester} requirements.",
                "The {semester} schedule includes advanced courses.",
                "Registration for {semester} begins next month.",
                "Academic calendar for {semester} is now available.",
                "Prerequisites for {semester} courses are listed below.",
                "The {semester} curriculum focuses on practical applications.",
                "Students enrolled in {semester} will study advanced topics.",
                "Course catalog for {semester} contains detailed information.",
                "The {semester} program offers specialized tracks."
            ]
            
            # Semester variations
            semester_variations = {
                'SEMESTER_1': ['first semester', 'semester 1', 'semester I'],
                'SEMESTER_2': ['second semester', 'semester 2', 'semester II'],
                'SEMESTER_3': ['third semester', 'semester 3', 'semester III'],
                'SEMESTER_4': ['fourth semester', 'semester 4', 'semester IV'],
                'FALL_SEMESTER': ['fall semester', 'fall term', 'autumn semester'],
                'SPRING_SEMESTER': ['spring semester', 'spring term'],
                'SUMMER_SEMESTER': ['summer semester', 'summer term', 'summer session'],
                'OTHER': ['general studies', 'elective courses', 'core curriculum']
            }
            
            # Generate samples
            for _ in range(num_samples):
                # Select random template and semester type
                template = np.random.choice(semester_templates)
                label = np.random.choice(list(semester_variations.keys()))
                semester_text = np.random.choice(semester_variations[label])
                
                # Generate text
                text = template.format(semester=semester_text)
                
                texts.append(text)
                labels.append(label)
            
            logger.info(f"Generated {len(texts)} synthetic training samples")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            return [], []


# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test text preprocessing
    sample_text = "This is the First Semester of Computer Science program. Students will study Fall Semester 2023 courses."
    
    print("Original text:", sample_text)
    print("Preprocessed text:", preprocessor.preprocess_text(sample_text))
    print("Extracted entities:", preprocessor.extract_semester_entities(sample_text))
    print("Features:", preprocessor.extract_features(sample_text))
    
    # Test synthetic data generation
    texts, labels = preprocessor.generate_synthetic_data(10)
    print(f"\nGenerated {len(texts)} synthetic samples:")
    for i in range(min(5, len(texts))):
        print(f"{i+1}. {texts[i]} -> {labels[i]}")