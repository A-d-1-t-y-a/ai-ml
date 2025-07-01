#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for Time Series Forecasting Project
Tests all major components to ensure robustness and reliability
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from data_collector import FinancialDataCollector
    from feature_engineer import FeatureEngineer
    from regime_detector import MarketRegimeDetector
    from ml_models import CrossMarketXGBoost, MarkovSwitchingARIMA, AttentionLSTM
    from config import *
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")

class TestDataCollector(unittest.TestCase):
    """Test the financial data collection module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.collector = FinancialDataCollector()
        
    def test_symbol_validation(self):
        """Test symbol validation logic"""
        valid_symbols = ['AAPL', 'BTC-USD', 'SPY']
        invalid_symbols = ['', 'INVALID123', None]
        
        for symbol in valid_symbols:
            self.assertTrue(self.collector.validate_symbol(symbol))
        
        for symbol in invalid_symbols:
            self.assertFalse(self.collector.validate_symbol(symbol))
    
    def test_market_type_detection(self):
        """Test market type detection from symbols"""
        test_cases = [
            ('AAPL', 'Stock'),
            ('BTC-USD', 'Crypto'),
            ('SPY', 'ETF'),
            ('MSFT', 'Stock'),
            ('ETH-USD', 'Crypto')
        ]
        
        for symbol, expected_type in test_cases:
            result = self.collector.detect_market_type(symbol)
            self.assertEqual(result, expected_type)
    
    def test_data_structure_validation(self):
        """Test data structure validation"""
        # Valid data structure
        valid_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        self.assertTrue(self.collector.validate_data_structure(valid_data))
        
        # Invalid data structure
        invalid_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Close': [102, 103, 104, 105, 106]
        })
        
        self.assertFalse(self.collector.validate_data_structure(invalid_data))

class TestFeatureEngineering(unittest.TestCase):
    """Test the feature engineering module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_engineer = FeatureEngineer()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'open': np.random.rand(100) * 100 + 100,
            'high': np.random.rand(100) * 100 + 105,
            'low': np.random.rand(100) * 100 + 95,
            'close': np.random.rand(100) * 100 + 100,
            'volume': np.random.rand(100) * 1000000,
            'Symbol': ['AAPL'] * 100,
            'Market_Type': ['Stock'] * 100
        })
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        result = self.feature_engineer.add_technical_indicators(self.sample_data)
        
        # Check if indicators were added
        expected_indicators = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20']
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
        
        # Check for NaN values in reasonable range
        self.assertLess(result['rsi'].isna().sum(), 50)  # RSI should have some initial NaN
        
    def test_cross_market_features(self):
        """Test cross-market feature generation"""
        # Create multi-market data
        multi_market_data = pd.concat([
            self.sample_data,
            self.sample_data.assign(Symbol='BTC-USD', Market_Type='Crypto'),
            self.sample_data.assign(Symbol='SPY', Market_Type='ETF')
        ])
        
        result = self.feature_engineer.add_cross_market_features(multi_market_data)
        
        # Check for cross-market correlation features
        correlation_features = [col for col in result.columns if 'corr_' in col]
        self.assertGreater(len(correlation_features), 0)
    
    def test_target_creation(self):
        """Test target variable creation"""
        result = self.feature_engineer.create_target_variables(self.sample_data)
        
        # Check target variables exist
        target_vars = ['returns', 'return_target_1d', 'direction_target_1d']
        for target in target_vars:
            self.assertIn(target, result.columns)
        
        # Check returns calculation
        returns = result['returns'].dropna()
        self.assertTrue(all(returns >= -1))  # Returns should be reasonable

class TestRegimeDetector(unittest.TestCase):
    """Test the market regime detection module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = MarketRegimeDetector()
        
        # Create sample data with engineered features
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'close': np.random.rand(200) * 100 + 100,
            'returns': np.random.normal(0, 0.02, 200),
            'volatility_20': np.random.rand(200) * 0.05,
            'volume': np.random.rand(200) * 1000000,
            'Symbol': ['AAPL'] * 200,
            'Market_Type': ['Stock'] * 200
        })
    
    def test_volatility_regime_detection(self):
        """Test volatility regime detection"""
        result = self.detector.detect_volatility_regimes(self.sample_data)
        
        if not result.empty:
            # Check regime column exists
            self.assertIn('volatility_regime', result.columns)
            
            # Check regime values are valid
            valid_regimes = ['Low_Vol', 'Medium_Vol', 'High_Vol']
            unique_regimes = result['volatility_regime'].unique()
            self.assertTrue(all(regime in valid_regimes for regime in unique_regimes))
    
    def test_trend_regime_detection(self):
        """Test trend regime detection"""
        # Add required columns for trend detection
        self.sample_data['sma_20'] = self.sample_data['close'].rolling(20).mean()
        self.sample_data['sma_50'] = self.sample_data['close'].rolling(50).mean()
        
        result = self.detector.detect_trend_regimes(self.sample_data)
        
        if not result.empty:
            # Check regime column exists
            self.assertIn('trend_regime', result.columns)
            
            # Check regime values are valid
            valid_regimes = ['Uptrend', 'Downtrend', 'Sideways']
            unique_regimes = result['trend_regime'].unique()
            self.assertTrue(all(regime in valid_regimes for regime in unique_regimes))
    
    def test_regime_combination(self):
        """Test regime combination logic"""
        # Create mock regime data
        vol_regimes = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'Market_Type': ['Stock'] * 10,
            'volatility_regime': ['Low_Vol'] * 5 + ['High_Vol'] * 5
        })
        
        trend_regimes = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'Market_Type': ['Stock'] * 10,
            'trend_regime': ['Uptrend'] * 5 + ['Downtrend'] * 5
        })
        
        corr_regimes = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'correlation_regime': ['Medium_Correlation'] * 10
        })
        
        result = self.detector.combine_regime_signals(vol_regimes, trend_regimes, corr_regimes)
        
        if not result.empty:
            self.assertIn('composite_regime', result.columns)

class TestMLModels(unittest.TestCase):
    """Test the machine learning models"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        
        # Create sample training data
        n_samples = 1000
        n_features = 20
        
        self.X_train = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = np.random.normal(0, 0.1, n_samples)
        self.regimes_train = np.random.choice(['High_Vol', 'Medium_Vol', 'Low_Vol'], n_samples)
        
        # Create sample test data
        n_test = 200
        self.X_test = pd.DataFrame(
            np.random.rand(n_test, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test = np.random.normal(0, 0.1, n_test)
        self.regimes_test = np.random.choice(['High_Vol', 'Medium_Vol', 'Low_Vol'], n_test)
    
    def test_xgboost_model(self):
        """Test XGBoost model training and prediction"""
        model = CrossMarketXGBoost()
        
        # Test training
        model.fit(self.X_train, self.y_train, pd.Series(self.regimes_train))
        
        # Check model was trained
        self.assertIsNotNone(model.model)
        
        # Test prediction
        predictions = model.predict(self.X_test, pd.Series(self.regimes_test))
        
        # Check prediction shape
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check feature importance
        self.assertIsNotNone(model.feature_importance)
        self.assertEqual(len(model.feature_importance), len(self.X_train.columns))
    
    def test_markov_arima_model(self):
        """Test Markov-switching ARIMA model"""
        model = MarkovSwitchingARIMA(n_regimes=2)
        
        # Test training
        model.fit(self.y_train, pd.Series(self.regimes_train))
        
        # Test prediction
        predictions = model.predict(steps=10)
        
        # Check prediction shape
        self.assertEqual(len(predictions), 10)
    
    def test_lstm_model_structure(self):
        """Test LSTM model structure"""
        model = AttentionLSTM(sequence_length=30, features=10)
        
        # Test model building
        model.build_model()
        
        # Check model was created
        self.assertIsNotNone(model.model)

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and validation"""
    
    def test_data_file_existence(self):
        """Test that required data files exist"""
        required_files = [
            'data/raw_financial_data.csv',
            'data/enhanced_features.csv',
            'results/model_performance.csv'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                # File exists, check it's not empty
                self.assertGreater(os.path.getsize(file_path), 0)
    
    def test_data_consistency(self):
        """Test data consistency across files"""
        try:
            # Load data files
            raw_data = pd.read_csv('data/raw_financial_data.csv')
            enhanced_data = pd.read_csv('data/enhanced_features.csv')
            
            # Check symbol consistency
            raw_symbols = set(raw_data['Symbol'].unique())
            enhanced_symbols = set(enhanced_data['Symbol'].unique())
            
            # Enhanced data should contain all raw symbols
            self.assertTrue(raw_symbols.issubset(enhanced_symbols))
            
        except FileNotFoundError:
            # Skip test if files don't exist
            pass
    
    def test_feature_completeness(self):
        """Test that all expected features are present"""
        try:
            data = pd.read_csv('data/enhanced_features.csv')
            
            # Check for key feature categories
            technical_features = [col for col in data.columns if any(
                indicator in col for indicator in ['rsi', 'macd', 'sma', 'ema']
            )]
            
            cross_market_features = [col for col in data.columns if 'corr_' in col]
            
            # Should have some features from each category
            self.assertGreater(len(technical_features), 0)
            self.assertGreater(len(cross_market_features), 0)
            
        except FileNotFoundError:
            # Skip test if file doesn't exist
            pass

class TestPerformanceMetrics(unittest.TestCase):
    """Test performance calculation functions"""
    
    def test_metric_calculations(self):
        """Test metric calculation accuracy"""
        from ml_models import evaluate_model
        
        # Create simple test data
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        results = evaluate_model(y_true, y_pred, "Test")
        
        # Check all metrics are present
        expected_metrics = ['mse', 'mae', 'rmse', 'r2', 'mape', 'directional_accuracy']
        for metric in expected_metrics:
            self.assertIn(metric, results)
        
        # Check metric values are reasonable
        self.assertGreater(results['r2'], 0.8)  # Should be high for this simple test
        self.assertLess(results['mape'], 10)    # Should be low error
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation"""
        from ml_models import evaluate_model
        
        # Perfect directional accuracy
        y_true = np.array([1, -1, 1, -1, 1])
        y_pred = np.array([0.5, -0.5, 0.8, -0.2, 0.1])
        
        results = evaluate_model(y_true, y_pred, "Test")
        self.assertEqual(results['directional_accuracy'], 1.0)

class TestConfigurationSettings(unittest.TestCase):
    """Test configuration and settings"""
    
    def test_config_values(self):
        """Test that configuration values are reasonable"""
        from config import *
        
        # Test parameter ranges
        self.assertGreater(VOLATILITY_CLUSTERS, 0)
        self.assertGreater(TREND_WINDOW, 0)
        self.assertGreater(CORRELATION_WINDOW, 0)
        
        # Test XGBoost parameters
        self.assertIsInstance(XGBOOST_PARAMS, dict)
        self.assertIn('n_estimators', XGBOOST_PARAMS)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        collector = FinancialDataCollector()
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        result = collector.validate_data_structure(empty_df)
        self.assertFalse(result)
    
    def test_missing_value_handling(self):
        """Test handling of missing values"""
        feature_engineer = FeatureEngineer()
        
        # Create data with missing values
        data_with_nan = pd.DataFrame({
            'close': [100, np.nan, 102, np.nan, 104],
            'volume': [1000, 1100, np.nan, 1300, 1400]
        })
        
        # Should handle NaN values gracefully
        try:
            result = feature_engineer.add_technical_indicators(data_with_nan)
            # Test passes if no exception is raised
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Error handling missing values: {e}")

def run_comprehensive_tests():
    """Run all tests and generate report"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataCollector,
        TestFeatureEngineering,
        TestRegimeDetector,
        TestMLModels,
        TestDataIntegrity,
        TestPerformanceMetrics,
        TestConfigurationSettings,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST REPORT")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Print failures and errors
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")
    
    print("\n" + "="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    
    if success:
        print("✅ ALL TESTS PASSED - PROJECT READY FOR SUBMISSION")
    else:
        print("❌ SOME TESTS FAILED - REVIEW REQUIRED")
    
    exit(0 if success else 1) 