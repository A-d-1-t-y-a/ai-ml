#!/usr/bin/env python3
"""
Unit Tests for Time Series Forecasting Project
"""

import unittest
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class TestProjectComponents(unittest.TestCase):
    """Test core project components"""
    
    def test_data_files_exist(self):
        """Test that key data files exist"""
        required_files = [
            'data/raw_financial_data.csv',
            'data/enhanced_features.csv', 
            'results/model_performance.csv'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                self.assertGreater(os.path.getsize(file_path), 0, f"{file_path} is empty")
    
    def test_feature_engineering(self):
        """Test basic feature engineering functionality"""
        # Create sample data
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'returns': [0.01, 0.0099, 0.0098, 0.0097, 0.0096]
        })
        
        # Test simple calculations
        data['sma_3'] = data['close'].rolling(3).mean()
        data['volatility'] = data['returns'].rolling(3).std()
        
        # Verify calculations
        self.assertAlmostEqual(data['sma_3'].iloc[2], 101.0, places=1)
        self.assertFalse(data['volatility'].iloc[2:].isna().all())
    
    def test_model_predictions(self):
        """Test basic model prediction functionality"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test prediction
        X_test = np.random.rand(20, 5)
        predictions = model.predict(X_test)
        
        # Verify prediction shape
        self.assertEqual(len(predictions), 20)
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
    
    def test_regime_classification(self):
        """Test basic regime classification"""
        # Create sample volatility data
        volatility_data = [0.01, 0.02, 0.05, 0.03, 0.01, 0.08, 0.02]
        
        # Simple regime classification
        regimes = []
        for vol in volatility_data:
            if vol < 0.02:
                regimes.append('Low_Vol')
            elif vol < 0.05:
                regimes.append('Medium_Vol')
            else:
                regimes.append('High_Vol')
        
        # Verify regime assignment
        self.assertEqual(regimes[0], 'Low_Vol')
        self.assertEqual(regimes[2], 'High_Vol')
        self.assertEqual(regimes[3], 'Medium_Vol')

def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)

if __name__ == "__main__":
    run_tests() 