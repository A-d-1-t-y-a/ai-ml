# Main Pipeline for Time Series Forecasting Project
"""
Complete Time Series Forecasting Pipeline with Cross-Market Analysis
This script orchestrates the entire machine learning pipeline from data collection to model deployment
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import *
import importlib.util

# Import renamed modules using importlib
def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import all required modules
data_collection = import_module_from_file("2_data_collection.py", "data_collection")
feature_engineering = import_module_from_file("3_feature_engineering.py", "feature_engineering")
regime_detection = import_module_from_file("4_regime_detection.py", "regime_detection")
ml_models = import_module_from_file("5_ml_models.py", "ml_models")

# Create shortcuts for classes
FinancialDataCollector = data_collection.FinancialDataCollector
FeatureEngineer = feature_engineering.FeatureEngineer
MarketRegimeDetector = regime_detection.MarketRegimeDetector
CrossMarketXGBoost = ml_models.CrossMarketXGBoost
prepare_model_data = ml_models.prepare_model_data
evaluate_model = ml_models.evaluate_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(f'{LOGS_DIR}pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TimeSeriesForecastingPipeline:
    """
    Complete pipeline for time series forecasting with regime detection and cross-market analysis
    """
    
    def __init__(self, s3_bucket=None):
        self.s3_bucket = s3_bucket
        self.data_collector = FinancialDataCollector(s3_bucket)
        self.feature_engineer = FeatureEngineer()
        self.regime_detector = MarketRegimeDetector()
        self.results = {}
        
    def run_data_collection(self):
        """Step 1: Collect financial data from multiple markets"""
        logger.info("="*60)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("="*60)
        
        # Collect data from all markets
        raw_data = self.data_collector.collect_all_markets()
        
        if raw_data.empty:
            logger.error("No data collected! Pipeline cannot continue.")
            return None
        
        # Save raw data
        self.data_collector.save_to_local(raw_data, 'raw_financial_data.csv')
        
        # Store data summary
        self.results['data_collection'] = {
            'total_records': len(raw_data),
            'date_range': f"{raw_data['date'].min()} to {raw_data['date'].max()}",
            'symbols': raw_data['Symbol'].nunique(),
            'markets': raw_data['Market_Type'].value_counts().to_dict()
        }
        
        logger.info(f"✓ Data collection completed: {len(raw_data):,} records")
        return raw_data
    
    def run_feature_engineering(self, raw_data):
        """Step 2: Engineer features including technical indicators and cross-market signals"""
        logger.info("="*60)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*60)
        
        # Process all features
        enhanced_data = self.feature_engineer.process_all_features(raw_data)
        
        if enhanced_data.empty:
            logger.error("Feature engineering failed! Pipeline cannot continue.")
            return None
        
        # Save enhanced data
        self.feature_engineer.save_to_local = lambda data, filename: data.to_csv(f"{DATA_DIR}{filename}", index=False)
        enhanced_data.to_csv(f"{DATA_DIR}enhanced_features.csv", index=False)
        
        # Store feature engineering summary
        self.results['feature_engineering'] = {
            'input_records': len(raw_data),
            'output_records': len(enhanced_data),
            'features_created': len(enhanced_data.columns) - len(raw_data.columns),
            'total_features': len(enhanced_data.columns)
        }
        
        logger.info(f"✓ Feature engineering completed: {len(enhanced_data.columns)} features")
        return enhanced_data
    
    def run_regime_detection(self, enhanced_data):
        """Step 3: Detect market regimes and create regime-based features"""
        logger.info("="*60)
        logger.info("STEP 3: MARKET REGIME DETECTION")
        logger.info("="*60)
        
        # Detect all regimes
        data_with_regimes, regime_summary = self.regime_detector.detect_all_regimes(enhanced_data)
        
        if data_with_regimes.empty:
            logger.warning("Regime detection failed, continuing without regime features")
            return enhanced_data
        
        # Save regime data
        data_with_regimes.to_csv(f"{DATA_DIR}data_with_regimes.csv", index=False)
        
        if not regime_summary.empty:
            regime_summary.to_csv(f"{DATA_DIR}regime_summary.csv", index=False)
        
        # Store regime detection summary
        if 'composite_regime' in data_with_regimes.columns:
            regime_dist = data_with_regimes['composite_regime'].value_counts().to_dict()
            self.results['regime_detection'] = {
                'regimes_detected': len(regime_dist),
                'regime_distribution': regime_dist
            }
        
        logger.info("✓ Regime detection completed")
        return data_with_regimes
    
    def run_model_training(self, final_data):
        """Step 4: Train machine learning models"""
        logger.info("="*60)
        logger.info("STEP 4: MODEL TRAINING & EVALUATION")
        logger.info("="*60)
        
        try:
            # Prepare data for modeling
            X_train, X_test, y_train, y_test, regimes_train, regimes_test = prepare_model_data(final_data)
            
            # Train XGBoost model
            logger.info("Training XGBoost model...")
            xgb_model = CrossMarketXGBoost()
            xgb_model.fit(X_train, y_train, regimes_train)
            
            # Make predictions
            xgb_predictions = xgb_model.predict(X_test, regimes_test)
            
            # Evaluate model
            xgb_results = evaluate_model(y_test, xgb_predictions)
            
            # Store model results
            self.results['model_performance'] = {
                'xgboost': xgb_results,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_used': len(X_train.columns)
            }
            
            # Save detailed results
            results_df = pd.DataFrame([xgb_results])
            results_df.to_csv(f"{RESULTS_DIR}model_performance.csv", index=False)
            
            # Save feature importance
            if xgb_model.feature_importance is not None:
                xgb_model.feature_importance.to_csv(f"{RESULTS_DIR}feature_importance.csv", index=False)
            
            logger.info("✓ Model training completed successfully")
            return xgb_model, xgb_results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return None, None
    
    def generate_report(self):
        """Generate comprehensive pipeline report"""
        logger.info("="*60)
        logger.info("GENERATING PIPELINE REPORT")
        logger.info("="*60)
        
        report = {
            'pipeline_timestamp': datetime.now().isoformat(),
            'pipeline_results': self.results
        }
        
        # Save report
        import json
        with open(f"{RESULTS_DIR}pipeline_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("TIME SERIES FORECASTING PIPELINE - FINAL REPORT")
        print("="*80)
        
        if 'data_collection' in self.results:
            dc = self.results['data_collection']
            print(f"\n📊 DATA COLLECTION:")
            print(f"   • Total Records: {dc['total_records']:,}")
            print(f"   • Date Range: {dc['date_range']}")
            print(f"   • Symbols: {dc['symbols']}")
            print(f"   • Markets: {dc['markets']}")
        
        if 'feature_engineering' in self.results:
            fe = self.results['feature_engineering']
            print(f"\n🔧 FEATURE ENGINEERING:")
            print(f"   • Input Records: {fe['input_records']:,}")
            print(f"   • Output Records: {fe['output_records']:,}")
            print(f"   • Features Created: {fe['features_created']}")
            print(f"   • Total Features: {fe['total_features']}")
        
        if 'regime_detection' in self.results:
            rd = self.results['regime_detection']
            print(f"\n🎯 REGIME DETECTION:")
            print(f"   • Regimes Detected: {rd['regimes_detected']}")
            print(f"   • Distribution: {rd['regime_distribution']}")
        
        if 'model_performance' in self.results:
            mp = self.results['model_performance']
            print(f"\n🤖 MODEL PERFORMANCE:")
            print(f"   • Training Samples: {mp['training_samples']:,}")
            print(f"   • Test Samples: {mp['test_samples']:,}")
            print(f"   • Features Used: {mp['features_used']}")
            
            if 'xgboost' in mp:
                xgb_perf = mp['xgboost']
                print(f"   • XGBoost RMSE: {xgb_perf['rmse']:.4f}")
                print(f"   • XGBoost MAE: {xgb_perf['mae']:.4f}")
                print(f"   • XGBoost R²: {xgb_perf['r2']:.4f}")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"   • Raw Data: {DATA_DIR}raw_financial_data.csv")
        print(f"   • Enhanced Features: {DATA_DIR}enhanced_features.csv")
        print(f"   • Regime Data: {DATA_DIR}data_with_regimes.csv")
        print(f"   • Model Performance: {RESULTS_DIR}model_performance.csv")
        print(f"   • Feature Importance: {RESULTS_DIR}feature_importance.csv")
        print(f"   • Pipeline Report: {RESULTS_DIR}pipeline_report.json")
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
    
    def run_complete_pipeline(self):
        """Run the complete time series forecasting pipeline"""
        logger.info("🚀 Starting Complete Time Series Forecasting Pipeline")
        logger.info(f"Pipeline started at: {datetime.now()}")
        
        try:
            # Step 1: Data Collection
            raw_data = self.run_data_collection()
            if raw_data is None:
                return False
            
            # Step 2: Feature Engineering
            enhanced_data = self.run_feature_engineering(raw_data)
            if enhanced_data is None:
                return False
            
            # Step 3: Regime Detection
            final_data = self.run_regime_detection(enhanced_data)
            
            # Step 4: Model Training
            model, results = self.run_model_training(final_data)
            
            # Step 5: Generate Report
            self.generate_report()
            
            logger.info("✅ Complete pipeline executed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return False

def main():
    """Main function to run the complete pipeline"""
    print("🎯 Time Series Forecasting Project: Cross-Market Predictive Analytics")
    print("📈 Integrating Market Regime Detection with Cross-Market Signal Analysis")
    print("🔬 Novel Approach: Multi-Market Regime-Aware Forecasting")
    print("\n" + "="*80)
    
    # Initialize pipeline
    pipeline = TimeSeriesForecastingPipeline()
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        print("🎉 Your time series forecasting system is ready for deployment!")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 