"""
Demo Script for Semester Named Entity Recognition System
Demonstrates all features with sample data and examples
"""

import os
import sys
import json
import time
from datetime import datetime

import config
from main import SemesterNERApplication
from train_model import create_sample_training_data

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title.center(78)} ")
    print("="*80)

def print_subheader(title):
    """Print a formatted subheader"""
    print(f"\n{title}")
    print("-" * len(title))

def demo_synthetic_training():
    """Demonstrate model training with synthetic data"""
    print_header("DEMO: MODEL TRAINING WITH SYNTHETIC DATA")
    
    app = SemesterNERApplication()
    
    print("Training Random Forest model with synthetic data...")
    print("This will generate 2000+ training samples automatically.")
    
    start_time = time.time()
    results = app.train_model(use_synthetic_data=True)
    training_time = time.time() - start_time
    
    print_subheader("Training Results")
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Validation Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation Accuracy: {results['cv_mean_accuracy']:.4f} (+/- {results['cv_std_accuracy']:.4f})")
    print(f"Training Samples: {results['training_samples']}")
    print(f"Number of Classes: {results['n_classes']}")
    print(f"Classes: {', '.join(results['class_names'])}")
    
    return app

def demo_text_predictions(app):
    """Demonstrate text predictions"""
    print_header("DEMO: TEXT PREDICTIONS")
    
    sample_texts = [
        "This course is offered in the first semester of the Computer Science program.",
        "Students must complete all spring semester requirements before graduation.",
        "The fall semester includes advanced mathematics and programming courses.",
        "Third semester students will focus on data structures and algorithms.",
        "Summer semester offers intensive programming bootcamp sessions.",
        "Fifth semester curriculum includes machine learning and AI courses.",
        "The winter term has special elective courses available.",
        "Eighth semester is the final semester with capstone projects.",
        "This is a general education requirement for all students.",
        "The academic year is divided into multiple terms and semesters."
    ]
    
    print("Making predictions on sample texts...")
    predictions = app.ner_model.predict(sample_texts)
    
    print_subheader("Prediction Results")
    for i, pred in enumerate(predictions, 1):
        print(f"\n{i}. Text: \"{pred['text']}\"")
        print(f"   Predicted Label: {pred['predicted_label']}")
        print(f"   Confidence: {pred['confidence']:.4f}")
        
        if pred['entities']:
            entities_info = []
            for entity in pred['entities']:
                entities_info.append(f"{entity['text']} ({entity['label']})")
            print(f"   Extracted Entities: {', '.join(entities_info)}")
        
        # Show top 3 probabilities
        sorted_probs = sorted(pred['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        prob_str = ', '.join([f"{label}: {prob:.3f}" for label, prob in sorted_probs])
        print(f"   Top Probabilities: {prob_str}")

def demo_pattern_matching():
    """Demonstrate pattern-based entity extraction"""
    print_header("DEMO: PATTERN-BASED ENTITY EXTRACTION")
    
    app = SemesterNERApplication()
    
    test_patterns = [
        "Students in semester 1 study foundational courses.",
        "The first semester includes mathematics and programming.",
        "Fall semester 2023 registration is now open.",
        "Spring term offers advanced electives.",
        "Summer semester has intensive workshop sessions.",
        "Academic year 2023-2024 semester guidelines.",
        "Freshman year first semester orientation.",
        "Second semester curriculum focuses on specialization.",
        "The 3rd semester includes research methodology.",
        "Final semester capstone project requirements."
    ]
    
    print("Extracting entities using pattern matching...")
    
    for i, text in enumerate(test_patterns, 1):
        entities = app.text_preprocessor.extract_semester_entities(text)
        features = app.text_preprocessor.extract_features(text)
        
        print(f"\n{i}. Text: \"{text}\"")
        
        if entities:
            for entity in entities:
                print(f"   Entity: '{entity['text']}' -> {entity['label']} (pattern: {entity['pattern_type']})")
        else:
            print("   No semester entities found")
        
        print(f"   Features: {features['semester_entity_count']} entities, "
              f"Keywords: {features['has_semester_keywords']}, "
              f"Numeric: {features['has_numeric_semester']}")

def create_sample_pdf_content():
    """Create sample PDF-like content for demonstration"""
    sample_content = """
    UNIVERSITY ACADEMIC CATALOG
    
    COMPUTER SCIENCE PROGRAM STRUCTURE
    
    The Computer Science program is designed as an 8-semester curriculum.
    
    FIRST SEMESTER:
    - Introduction to Programming
    - Mathematics I
    - English Composition
    - Computer Science Fundamentals
    
    SECOND SEMESTER:
    - Data Structures
    - Mathematics II
    - Physics I
    - Programming Laboratory
    
    SPRING SEMESTER OFFERINGS:
    - Advanced Programming
    - Database Systems
    - Software Engineering
    
    FALL SEMESTER SCHEDULE:
    - Algorithms and Complexity
    - Computer Networks
    - Operating Systems
    
    SUMMER SEMESTER:
    - Internship Program
    - Special Projects
    - Intensive Workshops
    
    FINAL SEMESTER (8th):
    - Capstone Project
    - Advanced Electives
    - Industry Seminar
    
    Students must complete all semester requirements for graduation.
    Each semester has specific prerequisites and credit requirements.
    """
    return sample_content

def demo_text_processing():
    """Demonstrate text processing and entity extraction"""
    print_header("DEMO: TEXT PROCESSING AND ENTITY EXTRACTION")
    
    app = SemesterNERApplication()
    
    # Train model if not already trained
    if not app.ner_model.is_trained:
        print("Training model first...")
        app.train_model(use_synthetic_data=True)
    
    sample_content = create_sample_pdf_content()
    
    print("Processing sample academic document content...")
    
    # Extract sentences
    sentences = app.text_preprocessor.extract_sentences(sample_content)
    print(f"\nExtracted {len(sentences)} sentences from the document")
    
    # Make predictions
    if sentences:
        predictions = app.ner_model.predict(sentences)
        
        # Filter high-confidence semester predictions
        semester_entities = [
            pred for pred in predictions 
            if pred['predicted_label'] != 'OTHER' and pred['confidence'] > 0.5
        ]
        
        print_subheader(f"Found {len(semester_entities)} High-Confidence Semester Entities")
        
        entity_counts = {}
        for entity in semester_entities:
            label = entity['predicted_label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
            
            print(f"\n• Text: \"{entity['text']}\"")
            print(f"  Label: {label}")
            print(f"  Confidence: {entity['confidence']:.4f}")
        
        print_subheader("Entity Distribution")
        for label, count in sorted(entity_counts.items()):
            print(f"{label}: {count} occurrences")
        
        # Calculate statistics
        if semester_entities:
            avg_confidence = sum(e['confidence'] for e in semester_entities) / len(semester_entities)
            print(f"\nAverage Confidence: {avg_confidence:.4f}")
            print(f"Total Semester Entities: {len(semester_entities)}")
            print(f"Coverage: {len(semester_entities)}/{len(sentences)} sentences ({100*len(semester_entities)/len(sentences):.1f}%)")

def demo_model_evaluation():
    """Demonstrate model evaluation with test data"""
    print_header("DEMO: MODEL EVALUATION")
    
    app = SemesterNERApplication()
    
    # Ensure model is trained
    if not app.ner_model.is_trained:
        print("Training model for evaluation...")
        app.train_model(use_synthetic_data=True)
    
    # Create test data
    test_data = [
        {"text": "First semester courses include programming fundamentals.", "label": "SEMESTER_1"},
        {"text": "Second semester focuses on data structures.", "label": "SEMESTER_2"},
        {"text": "Spring semester registration opens in January.", "label": "SPRING_SEMESTER"},
        {"text": "Fall semester includes advanced courses.", "label": "FALL_SEMESTER"},
        {"text": "Summer semester offers internship opportunities.", "label": "SUMMER_SEMESTER"},
        {"text": "Third semester curriculum covers algorithms.", "label": "SEMESTER_3"},
        {"text": "This is a general education requirement.", "label": "OTHER"},
        {"text": "Elective courses are available year-round.", "label": "OTHER"},
        {"text": "Fourth semester includes capstone preparation.", "label": "SEMESTER_4"},
        {"text": "Core curriculum requirements for all students.", "label": "OTHER"}
    ]
    
    test_texts = [item['text'] for item in test_data]
    test_labels = [item['label'] for item in test_data]
    
    print(f"Evaluating model on {len(test_data)} test samples...")
    
    # Make predictions
    predictions = app.ner_model.predict(test_texts)
    
    # Calculate accuracy manually for demo
    correct = 0
    for i, pred in enumerate(predictions):
        if pred['predicted_label'] == test_labels[i]:
            correct += 1
    
    accuracy = correct / len(test_data)
    
    print_subheader("Evaluation Results")
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{len(test_data)})")
    
    print_subheader("Detailed Predictions")
    for i, (pred, true_label) in enumerate(zip(predictions, test_labels)):
        status = "✓" if pred['predicted_label'] == true_label else "✗"
        print(f"\n{i+1}. {status} Text: \"{pred['text']}\"")
        print(f"   True: {true_label} | Predicted: {pred['predicted_label']} (conf: {pred['confidence']:.4f})")

def demo_synthetic_data_generation():
    """Demonstrate synthetic data generation"""
    print_header("DEMO: SYNTHETIC DATA GENERATION")
    
    app = SemesterNERApplication()
    
    print("Generating synthetic training data...")
    texts, labels = app.text_preprocessor.generate_synthetic_data(20)
    
    print(f"\nGenerated {len(texts)} synthetic samples:")
    
    # Show distribution of labels
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print_subheader("Label Distribution")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count} samples")
    
    print_subheader("Sample Generated Data")
    for i, (text, label) in enumerate(zip(texts[:10], labels[:10]), 1):
        print(f"{i}. \"{text}\" -> {label}")

def main():
    """Run the complete demo"""
    print_header("SEMESTER NAMED ENTITY RECOGNITION SYSTEM - DEMO")
    print("This demo showcases all features of the NER system")
    print("Estimated runtime: 2-3 minutes")
    
    start_time = time.time()
    
    try:
        # Demo 1: Synthetic Data Generation
        demo_synthetic_data_generation()
        
        # Demo 2: Model Training
        app = demo_synthetic_training()
        
        # Demo 3: Text Predictions
        demo_text_predictions(app)
        
        # Demo 4: Pattern Matching
        demo_pattern_matching()
        
        # Demo 5: Text Processing
        demo_text_processing()
        
        # Demo 6: Model Evaluation
        demo_model_evaluation()
        
        total_time = time.time() - start_time
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print(f"Total demo runtime: {total_time:.2f} seconds")
        print("\nThe system has demonstrated:")
        print("✓ Synthetic data generation")
        print("✓ Random Forest model training")
        print("✓ Text prediction with confidence scores")
        print("✓ Pattern-based entity extraction")
        print("✓ Document text processing")
        print("✓ Model evaluation and accuracy measurement")
        print("\nNext steps:")
        print("1. Try processing your own PDF files")
        print("2. Train with your own data")
        print("3. Explore the interactive mode")
        print("\nRun 'python main.py' to start the interactive application!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        print("Please check the installation and requirements.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)