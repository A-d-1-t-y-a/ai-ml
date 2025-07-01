#!/usr/bin/env python3
"""
Complete Pipeline Script for Time Series Forecasting Project
Processes the full dataset through all steps: feature engineering, regime detection, and basic modeling
"""

import pandas as pd
import numpy as np
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

def main():
    """Main function to run the complete pipeline"""
    print("🚀 Time Series Forecasting - Complete Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Load raw data
        print("\n📊 Step 1: Loading data...")
        data = pd.read_csv('data/raw_financial_data.csv')
        data['date'] = pd.to_datetime(data['date'])
        print(f"✅ Loaded {len(data):,} records from {data['Symbol'].nunique()} symbols")
        
        # Step 2: Feature Engineering
        print("\n🔧 Step 2: Feature Engineering...")
        from feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        enhanced_data = fe.process_all_features(data)
        enhanced_data.to_csv('data/enhanced_features.csv', index=False)
        print(f"✅ Created {len(enhanced_data.columns)} features, saved to enhanced_features.csv")
        
        # Step 3: Regime Detection
        print("\n🎯 Step 3: Regime Detection...")
        from regime_detector import MarketRegimeDetector
        detector = MarketRegimeDetector()
        regime_data, regime_summary = detector.detect_all_regimes(enhanced_data)
        regime_data.to_csv('data/data_with_regimes.csv', index=False)
        
        if not regime_summary.empty:
            regime_summary.to_csv('data/regime_summary.csv', index=False)
        
        if 'composite_regime' in regime_data.columns:
            regime_counts = regime_data['composite_regime'].value_counts()
            print(f"✅ Regime detection complete:")
            for regime, count in regime_counts.items():
                print(f"   {regime}: {count:,} records ({count/len(regime_data)*100:.1f}%)")
        else:
            print("✅ Regime detection complete (no explicit regime column)")
        
        # Step 4: Basic Model Training
        print("\n🤖 Step 4: Basic Model Training...")
        from ml_models import CrossMarketXGBoost, prepare_model_data, evaluate_model
        
        # Prepare data for modeling
        model_data = regime_data.select_dtypes(include=[np.number]).fillna(0)
        
        # Create target if it doesn't exist
        if 'return_target_1d' not in model_data.columns:
            print("Creating simple return target...")
            for symbol in regime_data['Symbol'].unique():
                mask = regime_data['Symbol'] == symbol
                symbol_data = regime_data[mask].sort_values('date')
                returns = symbol_data['close'].pct_change().shift(-1)
                model_data.loc[mask, 'return_target_1d'] = returns
        
        # Remove rows with missing targets
        model_data = model_data.dropna(subset=['return_target_1d'])
        
        if len(model_data) > 1000:  # Only train if we have enough data
            # Prepare features and target
            feature_cols = [col for col in model_data.columns if col not in ['return_target_1d', 'price_target_1d', 'direction_target_1d', 'volatility_target_1d']]
            X = model_data[feature_cols]
            y = model_data['return_target_1d']
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print(f"   Training set: {len(X_train):,} samples")
            print(f"   Test set: {len(X_test):,} samples")
            print(f"   Features: {len(feature_cols)}")
            
            # Train model
            model = CrossMarketXGBoost()
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_test)
            metrics = evaluate_model(y_test, predictions, "XGBoost")
            
            # Save results
            import json
            with open('results/pipeline_results.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print("✅ Model training complete:")
            print(f"   RMSE: {metrics.get('rmse', 'N/A'):.6f}")
            print(f"   MAE: {metrics.get('mae', 'N/A'):.6f}")
            print(f"   R²: {metrics.get('r2', 'N/A'):.4f}")
            print(f"   Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.1%}")
        
        # Step 5: Summary
        print("\n📈 Step 5: Pipeline Summary")
        print("=" * 50)
        print(f"✅ Data processed: {len(regime_data):,} records")
        print(f"✅ Features created: {len(regime_data.columns)} total")
        print(f"✅ Files saved:")
        print(f"   - data/enhanced_features.csv")
        print(f"   - data/data_with_regimes.csv")
        print(f"   - results/pipeline_results.json")
        
        print("\n🎉 Pipeline completed successfully!")
        print("\n📚 Next steps:")
        print("   - Run dashboard: streamlit run dashboard.py")
        print("   - Run main pipeline: python main_pipeline.py")
        print("   - View results in results/ folder")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ Pipeline failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main() 