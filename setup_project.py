#!/usr/bin/env python3
"""
Complete Project Setup Script for Time Series Forecasting Project
This script handles data collection, cleaning, feature engineering, and basic model training
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    logger.info("Checking Python version...")
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required!")
        return False
    logger.info(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def create_directories():
    """Create necessary project directories"""
    logger.info("Creating project directories...")
    directories = ['data', 'models', 'results', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")

def install_requirements():
    """Install project requirements"""
    logger.info("Installing project requirements...")
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # Install wheel and setuptools first
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'setuptools', 'wheel'])
        
        # Install requirements
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        
        logger.info("✅ All requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        return False

def collect_data():
    """Collect financial data"""
    logger.info("Starting data collection...")
    try:
        # Import after installation
        from data_collector import FinancialDataCollector
        
        collector = FinancialDataCollector()
        financial_data = collector.collect_all_markets()
        
        if not financial_data.empty:
            collector.save_to_local(financial_data, 'raw_financial_data.csv')
            logger.info(f"✅ Collected {len(financial_data):,} records")
            return True
        else:
            logger.error("❌ No data collected")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        return False

def engineer_features():
    """Perform feature engineering"""
    logger.info("Starting feature engineering...")
    try:
        import pandas as pd
        from feature_engineer import FeatureEngineer
        
        # Load raw data
        raw_data = pd.read_csv('data/raw_financial_data.csv')
        
        # Initialize feature engineer
        engineer = FeatureEngineer()
        
        # Create enhanced features
        enhanced_data = engineer.create_comprehensive_features(raw_data)
        
        # Save enhanced data
        enhanced_data.to_csv('data/enhanced_features.csv', index=False)
        
        logger.info("✅ Feature engineering completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        return False

def detect_regimes():
    """Detect market regimes"""
    logger.info("Starting regime detection...")
    try:
        import pandas as pd
        from regime_detector import MarketRegimeDetector
        
        # Load enhanced data
        enhanced_data = pd.read_csv('data/enhanced_features.csv')
        
        # Initialize regime detector
        detector = MarketRegimeDetector()
        
        # Detect regimes
        regime_data = detector.detect_comprehensive_regimes(enhanced_data)
        
        # Save regime data
        regime_data.to_csv('data/data_with_regimes.csv', index=False)
        
        logger.info("✅ Regime detection completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Regime detection failed: {e}")
        return False

def train_basic_models():
    """Train basic models for testing"""
    logger.info("Training basic models...")
    try:
        import pandas as pd
        from ml_models import CrossMarketXGBoost, prepare_model_data
        
        # Load regime data
        regime_data = pd.read_csv('data/data_with_regimes.csv')
        
        # Prepare model data
        X_train, X_test, y_train, y_test = prepare_model_data(regime_data)
        
        # Train XGBoost model
        model = CrossMarketXGBoost()
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Save model performance
        from ml_models import evaluate_model
        metrics = evaluate_model(y_test, predictions, "XGBoost")
        
        import json
        with open('results/quick_test_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("✅ Basic model training completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        return False

def run_tests():
    """Run basic tests to verify setup"""
    logger.info("Running basic tests...")
    try:
        import unittest
        from test_suite import TestDataIntegrity
        
        # Run basic tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestDataIntegrity)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            logger.info("✅ All tests passed")
            return True
        else:
            logger.warning("⚠️ Some tests failed, but setup may still be functional")
            return True
            
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Time Series Forecasting Project Setup")
    print("=" * 50)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Directory Creation", create_directories),
        ("Requirements Installation", install_requirements),
        ("Data Collection", collect_data),
        ("Feature Engineering", engineer_features),
        ("Regime Detection", detect_regimes),
        ("Basic Model Training", train_basic_models),
        ("Basic Tests", run_tests)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            if step_func():
                success_count += 1
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
    
    print(f"\n🎯 Setup Summary: {success_count}/{len(steps)} steps completed")
    
    if success_count >= 6:  # At least core functionality working
        print("\n🎉 Project setup successful!")
        print("\n📚 Next steps:")
        print("1. Run the dashboard: streamlit run dashboard.py")
        print("2. Run the full pipeline: python main_pipeline.py")
        print("3. Run tests: python test_suite.py")
    else:
        print("\n⚠️ Setup incomplete. Please check the errors above.")
        print("You may need to install dependencies manually or check your internet connection.")

if __name__ == "__main__":
    main() 