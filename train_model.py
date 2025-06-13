"""
Training Script for Semester Named Entity Recognition Model
Provides easy training with various data sources and options
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime

import config
from main import SemesterNERApplication

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def create_sample_training_data():
    """
    Create sample training data for demonstration purposes
    """
    sample_data = [
        {"text": "This course is offered in the first semester.", "label": "SEMESTER_1"},
        {"text": "Students must complete spring semester requirements.", "label": "SPRING_SEMESTER"},
        {"text": "The fall semester schedule includes advanced courses.", "label": "FALL_SEMESTER"},
        {"text": "Second semester curriculum focuses on practical applications.", "label": "SEMESTER_2"},
        {"text": "Summer semester offers intensive programs.", "label": "SUMMER_SEMESTER"},
        {"text": "Third semester students will study advanced topics.", "label": "SEMESTER_3"},
        {"text": "Fourth semester includes project work.", "label": "SEMESTER_4"},
        {"text": "The winter term has special courses.", "label": "OTHER"},
        {"text": "Fifth semester focuses on specialization.", "label": "SEMESTER_5"},
        {"text": "Sixth semester includes internship programs.", "label": "SEMESTER_6"},
        {"text": "Seventh semester has research projects.", "label": "SEMESTER_7"},
        {"text": "Eighth semester is the final semester.", "label": "SEMESTER_8"},
        {"text": "This is a general studies course.", "label": "OTHER"},
        {"text": "Elective courses are available throughout the year.", "label": "OTHER"},
        {"text": "Core curriculum requirements must be met.", "label": "OTHER"}
    ]
    
    # Save sample data
    sample_file = os.path.join(config.DATA_DIR, 'sample_training_data.json')
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample training data created: {sample_file}")
    return sample_file


def train_with_options(args):
    """
    Train model with specified options
    """
    try:
        # Initialize application
        app = SemesterNERApplication()
        
        # Create sample data if no training data provided
        training_data_path = args.train_data
        if not training_data_path:
            logger.info("No training data provided. Creating sample data...")
            training_data_path = create_sample_training_data()
        
        # Training configuration
        print("\n" + "="*60)
        print("SEMESTER NER MODEL TRAINING")
        print("="*60)
        print(f"Training data: {training_data_path if training_data_path else 'Synthetic data only'}")
        print(f"Use synthetic data: {args.use_synthetic}")
        print(f"Output directory: {config.OUTPUT_DIR}")
        print("-"*60)
        
        # Start training
        start_time = datetime.now()
        training_results = app.train_model(
            training_data_path=training_data_path,
            use_synthetic_data=args.use_synthetic
        )
        end_time = datetime.now()
        
        # Display results
        print("\nTRAINING RESULTS:")
        print("-"*40)
        print(f"Training Time: {(end_time - start_time).total_seconds():.2f} seconds")
        print(f"Validation Accuracy: {training_results['accuracy']:.4f}")
        print(f"Cross-validation Accuracy: {training_results['cv_mean_accuracy']:.4f} (+/- {training_results['cv_std_accuracy']:.4f})")
        print(f"Training Samples: {training_results['training_samples']}")
        print(f"Validation Samples: {training_results['validation_samples']}")
        print(f"Number of Features: {training_results['n_features']}")
        print(f"Number of Classes: {training_results['n_classes']}")
        print(f"Classes: {', '.join(training_results['class_names'])}")
        print(f"Model saved to: {training_results['model_path']}")
        
        # Save detailed training report
        report = {
            'training_timestamp': datetime.now().isoformat(),
            'training_duration_seconds': (end_time - start_time).total_seconds(),
            'training_data_path': training_data_path,
            'use_synthetic_data': args.use_synthetic,
            'results': training_results
        }
        
        report_file = os.path.join(config.OUTPUT_DIR, f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nDetailed training report saved to: {report_file}")
        
        # Test with sample predictions
        if args.test_predictions:
            print("\nTEST PREDICTIONS:")
            print("-"*40)
            
            test_texts = [
                "This course is available in the first semester of the program.",
                "Spring semester registration opens next month.",
                "Students in their third semester study advanced mathematics.",
                "The fall semester includes practical workshops.",
                "Summer term offers accelerated courses.",
                "This is a general education requirement."
            ]
            
            predictions = app.ner_model.predict(test_texts)
            
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. Text: {pred['text']}")
                print(f"   Predicted: {pred['predicted_label']} (confidence: {pred['confidence']:.4f})")
                if pred['entities']:
                    entities_str = ', '.join([f"{e['text']} ({e['label']})" for e in pred['entities']])
                    print(f"   Entities: {entities_str}")
                print()
        
        print("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False


def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description="Train Semester NER Model")
    
    parser.add_argument(
        '--train-data', 
        help='Path to training data (JSON or CSV file, or directory with PDFs)'
    )
    
    parser.add_argument(
        '--use-synthetic', 
        action='store_true', 
        default=True,
        help='Use synthetic data augmentation (default: True)'
    )
    
    parser.add_argument(
        '--no-synthetic', 
        action='store_true',
        help='Disable synthetic data augmentation'
    )
    
    parser.add_argument(
        '--test-predictions',
        action='store_true',
        default=True,
        help='Test model with sample predictions after training (default: True)'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for models and results'
    )
    
    parser.add_argument(
        '--create-sample-data',
        action='store_true',
        help='Create sample training data and exit'
    )
    
    args = parser.parse_args()
    
    # Handle no-synthetic flag
    if args.no_synthetic:
        args.use_synthetic = False
    
    # Override output directory if provided
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Create sample data if requested
    if args.create_sample_data:
        sample_file = create_sample_training_data()
        print(f"Sample training data created: {sample_file}")
        return
    
    try:
        success = train_with_options(args)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 