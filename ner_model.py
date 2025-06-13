"""
Named Entity Recognition Model using Random Forest
Handles training, prediction, and evaluation of semester NER
"""

import os
import logging
import pickle
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import config
from text_preprocessor import TextPreprocessor

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class SemesterNERModel:
    """
    Random Forest-based Named Entity Recognition model for semester entities
    """
    
    def __init__(self):
        """
        Initialize the NER model with Random Forest classifier
        """
        try:
            # Initialize Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS,
                max_depth=config.MAX_DEPTH,
                min_samples_split=config.MIN_SAMPLES_SPLIT,
                min_samples_leaf=config.MIN_SAMPLES_LEAF,
                random_state=config.RANDOM_STATE,
                n_jobs=-1  # Use all available processors
            )
            
            # Initialize text preprocessor
            self.preprocessor = TextPreprocessor()
            
            # Model state
            self.is_trained = False
            self.feature_names = []
            self.class_names = []
            
            logger.info("Semester NER Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NER Model: {str(e)}")
            raise
    
    def train(self, texts: List[str], labels: List[str] = None, use_synthetic_data: bool = True) -> Dict[str, any]:
        """
        Train the NER model
        
        Args:
            texts (List[str]): Training texts
            labels (List[str]): Training labels (optional, will generate if not provided)
            use_synthetic_data (bool): Whether to augment with synthetic data
            
        Returns:
            Dict: Training results and metrics
        """
        try:
            logger.info("Starting model training...")
            
            # Prepare training data
            training_texts = texts.copy() if texts else []
            training_labels = labels.copy() if labels else []
            
            # Generate synthetic data if requested
            if use_synthetic_data:
                logger.info("Generating synthetic training data...")
                synthetic_texts, synthetic_labels = self.preprocessor.generate_synthetic_data(2000)
                training_texts.extend(synthetic_texts)
                training_labels.extend(synthetic_labels)
            
            # If no labels provided, generate them using pattern matching
            if not labels and texts:
                logger.info("Generating labels using pattern matching...")
                generated_labels = []
                for text in texts:
                    entities = self.preprocessor.extract_semester_entities(text)
                    if entities:
                        generated_labels.append(entities[0]['label'])
                    else:
                        generated_labels.append('OTHER')
                training_labels = generated_labels + (training_labels if use_synthetic_data else [])
            
            if not training_texts or not training_labels:
                raise ValueError("No training data available")
            
            # Create feature matrix and labels
            logger.info("Creating feature matrix...")
            X, y = self.preprocessor.create_training_data(training_texts, training_labels)
            
            if X.size == 0 or y.size == 0:
                raise ValueError("Failed to create feature matrix")
            
            # Store class names for later use
            self.class_names = self.preprocessor.label_encoder.classes_.tolist()
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE,
                stratify=y
            )
            
            logger.info(f"Training set size: {X_train.shape[0]}")
            logger.info(f"Validation set size: {X_val.shape[0]}")
            logger.info(f"Number of features: {X_train.shape[1]}")
            logger.info(f"Number of classes: {len(self.class_names)}")
            
            # Train the model
            logger.info("Training Random Forest model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Perform cross-validation
            logger.info("Performing cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=config.CROSS_VALIDATION_FOLDS,
                scoring='accuracy'
            )
            
            # Create detailed evaluation report
            classification_rep = classification_report(
                y_val, y_pred, 
                target_names=self.class_names,
                output_dict=True
            )
            
            # Calculate feature importance
            feature_importance = self.model.feature_importances_.argsort()[-20:][::-1]
            
            self.is_trained = True
            
            # Prepare training results
            training_results = {
                'accuracy': accuracy,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'classification_report': classification_rep,
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
                'feature_importance_indices': feature_importance.tolist(),
                'training_samples': len(training_texts),
                'validation_samples': X_val.shape[0],
                'n_features': X_train.shape[1],
                'n_classes': len(self.class_names),
                'class_names': self.class_names
            }
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Validation Accuracy: {accuracy:.4f}")
            logger.info(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict semester entities in texts
        
        Args:
            texts (List[str]): Input texts for prediction
            
        Returns:
            List[Dict]: Predictions with confidence scores
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if not texts:
                return []
            
            logger.info(f"Making predictions for {len(texts)} texts...")
            
            predictions = []
            
            for idx, text in enumerate(texts):
                # Extract features
                X = self.preprocessor.transform_text([text])
                
                if X.size == 0:
                    predictions.append({
                        'text': text,
                        'predicted_label': 'OTHER',
                        'confidence': 0.0,
                        'all_probabilities': {},
                        'entities': []
                    })
                    continue
                
                # Make prediction
                pred_label_idx = self.model.predict(X)[0]
                pred_probabilities = self.model.predict_proba(X)[0]
                
                # Get predicted label
                predicted_label = self.class_names[pred_label_idx]
                confidence = pred_probabilities[pred_label_idx]
                
                # Create probability dictionary
                prob_dict = {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(pred_probabilities)
                }
                
                # Extract entities using pattern matching
                entities = self.preprocessor.extract_semester_entities(text)
                
                prediction = {
                    'text': text,
                    'predicted_label': predicted_label,
                    'confidence': float(confidence),
                    'all_probabilities': prob_dict,
                    'entities': entities,
                    'text_features': self.preprocessor.extract_features(text)
                }
                
                predictions.append(prediction)
            
            logger.info(f"Completed predictions for {len(texts)} texts")
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def predict_single(self, text: str) -> Dict[str, any]:
        """
        Predict semester entities in a single text
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Prediction result
        """
        predictions = self.predict([text])
        return predictions[0] if predictions else {}
    
    def evaluate(self, test_texts: List[str], test_labels: List[str]) -> Dict[str, any]:
        """
        Evaluate model performance on test data
        
        Args:
            test_texts (List[str]): Test texts
            test_labels (List[str]): True labels
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            logger.info(f"Evaluating model on {len(test_texts)} test samples...")
            
            # Create test feature matrix
            X_test, y_test = self.preprocessor.create_training_data(test_texts, test_labels)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(
                y_test, y_pred,
                target_names=self.class_names,
                output_dict=True
            )
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            evaluation_results = {
                'accuracy': accuracy,
                'classification_report': classification_rep,
                'confusion_matrix': conf_matrix.tolist(),
                'test_samples': len(test_texts),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'true_labels': y_test.tolist()
            }
            
            logger.info(f"Evaluation completed. Test Accuracy: {accuracy:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def save_model(self, model_path: str = None) -> str:
        """
        Save the trained model to disk
        
        Args:
            model_path (str): Path to save the model (optional)
            
        Returns:
            str: Path where model was saved
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            # Use default path if not provided
            if not model_path:
                model_path = os.path.join(config.MODELS_DIR, config.MODEL_NAME)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model components
            model_data = {
                'model': self.model,
                'preprocessor': self.preprocessor,
                'is_trained': self.is_trained,
                'class_names': self.class_names,
                'feature_names': self.feature_names
            }
            
            # Save using joblib for sklearn objects
            joblib.dump(model_data, model_path)
            
            logger.info(f"Model saved successfully to: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load a trained model from disk
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Use default path if not provided
            if not model_path:
                model_path = os.path.join(config.MODELS_DIR, config.MODEL_NAME)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model components
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.is_trained = model_data['is_trained']
            self.class_names = model_data['class_names']
            self.feature_names = model_data.get('feature_names', [])
            
            logger.info(f"Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            Dict: Best parameters and scores
        """
        try:
            logger.info("Starting hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.4f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return tuning_results
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            return {}
    
    def visualize_results(self, evaluation_results: Dict[str, any], save_path: str = None):
        """
        Create visualizations of model performance
        
        Args:
            evaluation_results (Dict): Results from model evaluation
            save_path (str): Path to save plots (optional)
        """
        try:
            # Create confusion matrix heatmap
            plt.figure(figsize=(10, 8))
            conf_matrix = np.array(evaluation_results['confusion_matrix'])
            
            sns.heatmap(
                conf_matrix, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            if save_path:
                confusion_path = os.path.join(save_path, 'confusion_matrix.png')
                plt.savefig(confusion_path)
                logger.info(f"Confusion matrix saved to: {confusion_path}")
            
            plt.show()
            
            # Create feature importance plot if available
            if hasattr(self.model, 'feature_importances_'):
                plt.figure(figsize=(12, 6))
                
                # Get top 20 most important features
                feature_importance = self.model.feature_importances_
                top_indices = feature_importance.argsort()[-20:][::-1]
                top_importance = feature_importance[top_indices]
                
                plt.bar(range(len(top_importance)), top_importance)
                plt.title('Top 20 Feature Importances')
                plt.xlabel('Feature Index')
                plt.ylabel('Importance')
                plt.tight_layout()
                
                if save_path:
                    importance_path = os.path.join(save_path, 'feature_importance.png')
                    plt.savefig(importance_path)
                    logger.info(f"Feature importance plot saved to: {importance_path}")
                
                plt.show()
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the current model
        
        Returns:
            Dict: Model information
        """
        return {
            'is_trained': self.is_trained,
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.model.n_estimators if hasattr(self.model, 'n_estimators') else None,
            'max_depth': self.model.max_depth if hasattr(self.model, 'max_depth') else None,
            'n_classes': len(self.class_names),
            'class_names': self.class_names,
            'n_features': len(self.feature_names) if self.feature_names else None
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    ner_model = SemesterNERModel()
    
    # Generate sample training data
    sample_texts = [
        "This course is offered in the first semester.",
        "Students must complete spring semester requirements.",
        "The fall semester schedule includes advanced courses.",
        "Second semester curriculum focuses on practical applications.",
        "Summer semester offers intensive programs.",
        "This is a general studies course."
    ]
    
    # Train model
    try:
        training_results = ner_model.train(sample_texts, use_synthetic_data=True)
        
        print("Training Results:")
        print(f"Accuracy: {training_results['accuracy']:.4f}")
        print(f"CV Accuracy: {training_results['cv_mean_accuracy']:.4f}")
        print(f"Classes: {training_results['class_names']}")
        
        # Test predictions
        test_texts = [
            "Registration for first semester begins next month.",
            "The spring semester program offers specialized tracks.",
            "This is an elective course available throughout the year."
        ]
        
        predictions = ner_model.predict(test_texts)
        
        print("\nPredictions:")
        for pred in predictions:
            print(f"Text: {pred['text']}")
            print(f"Predicted: {pred['predicted_label']} (confidence: {pred['confidence']:.4f})")
            print(f"Entities: {pred['entities']}")
            print("-" * 50)
        
        # Save model
        model_path = ner_model.save_model()
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")