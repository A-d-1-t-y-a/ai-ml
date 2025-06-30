# ML Models Module for Time Series Forecasting Project
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Optional TensorFlow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available, LSTM models will be disabled")

from config import *

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class CrossMarketXGBoost:
    """XGBoost model with cross-market features"""
    
    def __init__(self, **params):
        self.params = {**XGBOOST_PARAMS, **params}
        self.model = None
        self.feature_importance = None
        
    def prepare_features(self, X, regimes=None):
        """Prepare features for XGBoost training"""
        features = X.copy()
        
        if regimes is not None:
            regime_dummies = pd.get_dummies(regimes, prefix='regime')
            features = pd.concat([features, regime_dummies], axis=1)
        
        numeric_features = features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(0)
        
        return numeric_features
    
    def fit(self, X, y, regimes=None):
        """Train XGBoost model"""
        logger.info("Training Cross-Market XGBoost model...")
        
        X_prepared = self.prepare_features(X, regimes)
        
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_prepared, y)
        
        self.feature_importance = pd.DataFrame({
            'feature': X_prepared.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"XGBoost model trained with {len(X_prepared.columns)} features")
    
    def predict(self, X, regimes=None):
        """Make predictions"""
        X_prepared = self.prepare_features(X, regimes)
        return self.model.predict(X_prepared)

def prepare_model_data(df, target_column='return_target_1d', test_size=0.2):
    """Prepare data for model training"""
    logger.info("Preparing data for model training...")
    
    df_clean = df.dropna(subset=[target_column]).copy()
    
    feature_columns = [col for col in df_clean.columns 
                      if col not in ['date', 'Symbol', 'Market_Type', target_column] 
                      and not col.startswith('target')]
    
    X = df_clean[feature_columns]
    y = df_clean[target_column]
    regimes = df_clean.get('composite_regime', pd.Series(['Unknown'] * len(df_clean)))
    
    split_index = int(len(df_clean) * (1 - test_size))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    regimes_train = regimes.iloc[:split_index]
    regimes_test = regimes.iloc[split_index:]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, regimes_train, regimes_test

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    """Main function to train and evaluate models"""
    logger.info("Starting ML Model Training...")
    
    try:
        data = pd.read_csv(f"{DATA_DIR}data_with_regimes.csv")
        data['date'] = pd.to_datetime(data['date'])
        
        X_train, X_test, y_train, y_test, regimes_train, regimes_test = prepare_model_data(data)
        
        xgb_model = CrossMarketXGBoost()
        xgb_model.fit(X_train, y_train, regimes_train)
        
        xgb_predictions = xgb_model.predict(X_test, regimes_test)
        xgb_results = evaluate_model(y_test, xgb_predictions)
        
        print("\nMODEL EVALUATION RESULTS")
        print("="*40)
        print("XGBoost Model:")
        print(f"  RMSE: {xgb_results['rmse']:.4f}")
        print(f"  MAE:  {xgb_results['mae']:.4f}")
        print(f"  R²:   {xgb_results['r2']:.4f}")
        
        results_df = pd.DataFrame([xgb_results])
        results_df.to_csv(f"{RESULTS_DIR}model_performance.csv", index=False)
        
        if xgb_model.feature_importance is not None:
            xgb_model.feature_importance.to_csv(f"{RESULTS_DIR}feature_importance.csv", index=False)
            print(f"\nTop 10 Important Features:")
            print(xgb_model.feature_importance.head(10))
        
        logger.info("Model training completed successfully")
        
    except FileNotFoundError:
        logger.error("Data file not found. Please run previous steps first.")
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")

if __name__ == "__main__":
    main()
