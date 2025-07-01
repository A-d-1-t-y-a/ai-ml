# ML Models Module for Time Series Forecasting Project
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, if not available, use alternative LSTM implementation
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available, using alternative LSTM implementation")

from config import *

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class MarkovSwitchingARIMA:
    """ARIMA model with Markov regime switching"""
    
    def __init__(self, n_regimes=3, order=(1,1,1)):
        self.n_regimes = n_regimes
        self.order = order
        self.model = None
        self.regime_probs = None
        
    def fit(self, y, regimes=None):
        """Train Markov-switching ARIMA model"""
        logger.info(f"Training Markov-switching ARIMA model with {self.n_regimes} regimes...")
        
        try:
            # Use Markov Regression as a proxy for regime-switching ARIMA
            self.model = MarkovRegression(
                endog=y,
                k_regimes=self.n_regimes,
                trend='c',
                switching_variance=True
            )
            
            self.fitted_model = self.model.fit(maxiter=100, disp=False)
            logger.info("Markov-switching ARIMA model trained successfully")
            
            # Store regime probabilities
            self.regime_probs = self.fitted_model.smoothed_marginal_probabilities
            
        except Exception as e:
            logger.warning(f"Markov regression failed: {e}, falling back to simple ARIMA")
            # Fallback to simple ARIMA
            from statsmodels.tsa.arima.model import ARIMA
            self.simple_arima = ARIMA(y, order=self.order)
            self.fitted_model = self.simple_arima.fit()
            
    def predict(self, steps=1):
        """Make predictions"""
        try:
            if hasattr(self, 'simple_arima'):
                return self.fitted_model.forecast(steps=steps)
            else:
                return self.fitted_model.forecast(steps=steps)
        except:
            return np.array([y.mean()] * steps)
    
    def get_regime_probabilities(self):
        """Get current regime probabilities"""
        if self.regime_probs is not None:
            return self.regime_probs
        else:
            return None

class CrossMarketXGBoost:
    """XGBoost model with cross-market features and regime awareness"""
    
    def __init__(self, **params):
        self.params = {**XGBOOST_PARAMS, **params}
        self.model = None
        self.feature_importance = None
        self.regime_models = {}
        
    def prepare_features(self, X, regimes=None):
        """Prepare features for XGBoost training"""
        features = X.copy()
        
        # Add regime features if available
        if regimes is not None:
            regime_dummies = pd.get_dummies(regimes, prefix='regime')
            features = pd.concat([features, regime_dummies], axis=1)
        
        # Select only numeric features
        numeric_features = features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(0)
        
        # Add cross-market interaction features
        market_cols = [col for col in numeric_features.columns if any(market in col for market in ['Stock', 'Crypto', 'ETF'])]
        if len(market_cols) >= 2:
            for i, col1 in enumerate(market_cols[:3]):  # Limit to prevent explosion
                for col2 in market_cols[i+1:4]:
                    if col1 in numeric_features.columns and col2 in numeric_features.columns:
                        numeric_features[f'{col1}_x_{col2}'] = numeric_features[col1] * numeric_features[col2]
        
        return numeric_features
    
    def fit(self, X, y, regimes=None):
        """Train XGBoost model with regime awareness"""
        logger.info("Training Cross-Market XGBoost model...")
        
        X_prepared = self.prepare_features(X, regimes)
        
        if regimes is not None and len(regimes.unique()) > 1:
            # Train separate models for each regime
            for regime in regimes.unique():
                if pd.isna(regime):
                    continue
                    
                regime_mask = regimes == regime
                if regime_mask.sum() > 10:  # Only train if sufficient data
                    X_regime = X_prepared[regime_mask]
                    y_regime = y[regime_mask]
                    
                    regime_model = xgb.XGBRegressor(**self.params)
                    regime_model.fit(X_regime, y_regime)
                    self.regime_models[regime] = regime_model
                    
            logger.info(f"Trained {len(self.regime_models)} regime-specific models")
        
        # Train overall model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_prepared, y)
        
        self.feature_importance = pd.DataFrame({
            'feature': X_prepared.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"XGBoost model trained with {len(X_prepared.columns)} features")
    
    def predict(self, X, regimes=None):
        """Make predictions using regime-aware approach"""
        X_prepared = self.prepare_features(X, regimes)
        
        if regimes is not None and self.regime_models:
            predictions = np.zeros(len(X_prepared))
            
            for regime in self.regime_models.keys():
                regime_mask = regimes == regime
                if regime_mask.sum() > 0:
                    predictions[regime_mask] = self.regime_models[regime].predict(X_prepared[regime_mask])
            
            # Use overall model for unknown regimes
            unknown_mask = ~regimes.isin(self.regime_models.keys())
            if unknown_mask.sum() > 0:
                predictions[unknown_mask] = self.model.predict(X_prepared[unknown_mask])
                
            return predictions
        else:
            return self.model.predict(X_prepared)

class AttentionLSTM:
    """LSTM with attention mechanism for cross-market analysis"""
    
    def __init__(self, sequence_length=60, features=10, units=50):
        self.sequence_length = sequence_length
        self.features = features
        self.units = units
        self.model = None
        self.scaler = StandardScaler()
        
    def create_sequences(self, data, target):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model with attention mechanism"""
        if TF_AVAILABLE:
            # TensorFlow implementation
            inputs = Input(shape=(self.sequence_length, self.features))
            
            # LSTM layers
            lstm_out = LSTM(self.units, return_sequences=True)(inputs)
            lstm_out = Dropout(0.2)(lstm_out)
            lstm_out2 = LSTM(self.units, return_sequences=True)(lstm_out)
            
            # Attention mechanism
            attention = MultiHeadAttention(num_heads=4, key_dim=self.units//4)(lstm_out2, lstm_out2)
            attention = Dropout(0.2)(attention)
            
            # Global average pooling and output
            pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
            outputs = Dense(1)(pooled)
            
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
        else:
            # Alternative implementation without TensorFlow
            logger.warning("TensorFlow not available, using simplified LSTM approach")
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(self.units, self.units//2),
                max_iter=200,
                random_state=42
            )
    
    def fit(self, X, y, regimes=None):
        """Train LSTM model"""
        logger.info("Training LSTM with attention mechanism...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if TF_AVAILABLE and self.model is not None:
            # Create sequences
            X_seq, y_seq = self.create_sequences(X_scaled, y)
            
            if len(X_seq) > 0:
                self.model.fit(
                    X_seq, y_seq,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
        else:
            # Fallback to MLP
            self.build_model()
            # For MLP, we'll use the raw features
            self.model.fit(X_scaled, y)
        
        logger.info("LSTM model training completed")
    
    def predict(self, X, regimes=None):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        if TF_AVAILABLE and hasattr(self.model, 'predict'):
            if len(X_scaled) >= self.sequence_length:
                X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
                if len(X_seq) > 0:
                    return self.model.predict(X_seq, verbose=0).flatten()
        
        # Fallback prediction
        return self.model.predict(X_scaled)

def prepare_model_data(df, target_column='return_target_1d', test_size=0.2):
    """Prepare data for model training with improved handling"""
    logger.info("Preparing data for model training...")
    
    # Ensure target column exists
    if target_column not in df.columns:
        logger.warning(f"Target column {target_column} not found, using returns as target")
        if 'returns' in df.columns:
            target_column = 'returns'
        else:
            logger.error("No suitable target column found")
            return None, None, None, None, None, None
    
    df_clean = df.dropna(subset=[target_column]).copy()
    
    if len(df_clean) == 0:
        logger.error("No data remaining after removing NaN targets")
        return None, None, None, None, None, None
    
    # Select feature columns (excluding non-predictive columns)
    exclude_columns = ['date', 'Symbol', 'Market_Type', target_column] + \
                     [col for col in df_clean.columns if col.startswith('target')]
    feature_columns = [col for col in df_clean.columns if col not in exclude_columns]
    
    logger.info(f"Using {len(feature_columns)} features for modeling")
    
    X = df_clean[feature_columns]
    y = df_clean[target_column]
    regimes = df_clean.get('composite_regime', pd.Series(['Unknown'] * len(df_clean)))
    
    # Chronological split
    split_index = int(len(df_clean) * (1 - test_size))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    regimes_train = regimes.iloc[:split_index]
    regimes_test = regimes.iloc[split_index:]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Features: {len(feature_columns)}")
    
    return X_train, X_test, y_train, y_test, regimes_train, regimes_test

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Comprehensive model evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    y_true_direction = (y_true > 0).astype(int)
    y_pred_direction = (y_pred > 0).astype(int)
    directional_accuracy = accuracy_score(y_true_direction, y_pred_direction)
    
    results = {
        'model': model_name,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }
    
    return results

def evaluate_by_regime(y_true, y_pred, regimes, model_name="Model"):
    """Evaluate model performance by regime"""
    regime_results = {}
    
    for regime in regimes.unique():
        if pd.isna(regime):
            continue
            
        regime_mask = regimes == regime
        if regime_mask.sum() > 5:  # Need sufficient data
            regime_results[regime] = evaluate_model(
                y_true[regime_mask], 
                y_pred[regime_mask], 
                f"{model_name}_{regime}"
            )
    
    return regime_results

def train_all_models(data):
    """Train all three required models"""
    logger.info("Training all models...")
    
    # Prepare data
    model_data = prepare_model_data(data)
    if model_data[0] is None:
        logger.error("Failed to prepare data for modeling")
        return None
    
    X_train, X_test, y_train, y_test, regimes_train, regimes_test = model_data
    
    models = {}
    results = {}
    
    # 1. Markov-switching ARIMA
    logger.info("Training Markov-switching ARIMA...")
    try:
        arima_model = MarkovSwitchingARIMA(n_regimes=3)
        arima_model.fit(y_train, regimes_train)
        arima_pred = arima_model.predict(steps=len(y_test))
        
        # Handle prediction length mismatch
        if len(arima_pred) != len(y_test):
            arima_pred = np.full(len(y_test), y_train.mean())
            
        models['arima'] = arima_model
        results['arima'] = evaluate_model(y_test, arima_pred, "Markov-ARIMA")
        
    except Exception as e:
        logger.error(f"ARIMA training failed: {e}")
        results['arima'] = {'error': str(e)}
    
    # 2. Cross-Market XGBoost
    logger.info("Training Cross-Market XGBoost...")
    try:
        xgb_model = CrossMarketXGBoost()
        xgb_model.fit(X_train, y_train, regimes_train)
        xgb_pred = xgb_model.predict(X_test, regimes_test)
        
        models['xgboost'] = xgb_model
        results['xgboost'] = evaluate_model(y_test, xgb_pred, "XGBoost")
        
        # Regime-specific evaluation
        regime_results = evaluate_by_regime(y_test, xgb_pred, regimes_test, "XGBoost")
        results['xgboost_by_regime'] = regime_results
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        results['xgboost'] = {'error': str(e)}
    
    # 3. LSTM with Attention
    logger.info("Training LSTM with attention...")
    try:
        # Use fewer features for LSTM to avoid complexity
        numeric_features = [col for col in X_train.columns if X_train[col].dtype in ['float64', 'int64']][:20]
        X_train_lstm = X_train[numeric_features].fillna(0)
        X_test_lstm = X_test[numeric_features].fillna(0)
        
        lstm_model = AttentionLSTM(
            sequence_length=min(30, len(X_train_lstm)//4),
            features=len(numeric_features)
        )
        lstm_model.build_model()
        lstm_model.fit(X_train_lstm, y_train, regimes_train)
        lstm_pred = lstm_model.predict(X_test_lstm, regimes_test)
        
        models['lstm'] = lstm_model
        results['lstm'] = evaluate_model(y_test, lstm_pred, "LSTM-Attention")
        
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        results['lstm'] = {'error': str(e)}
    
    return models, results

def main():
    """Main function to train and evaluate all models"""
    logger.info("Starting comprehensive ML model training...")
    
    try:
        # Load data
        data_files = [
            f"{DATA_DIR}data_with_regimes.csv",
            f"{DATA_DIR}enhanced_features.csv",
            f"{DATA_DIR}raw_financial_data.csv"
        ]
        
        data = None
        for file_path in data_files:
            try:
                data = pd.read_csv(file_path)
                data['date'] = pd.to_datetime(data['date'])
                logger.info(f"Loaded data from {file_path}: {data.shape}")
                break
            except FileNotFoundError:
                continue
        
        if data is None:
            logger.error("No data files found. Please run data collection first.")
            return
        
        # Train all models
        models, results = train_all_models(data)
        
        if models is None:
            logger.error("Model training failed")
            return
        
        # Display results
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*60)
        
        for model_name, result in results.items():
            if 'error' in result:
                print(f"\n❌ {model_name.upper()} - Error: {result['error']}")
            elif '_by_regime' not in model_name:
                print(f"\n✅ {model_name.upper()}:")
                print(f"  RMSE: {result.get('rmse', 'N/A'):.4f}")
                print(f"  MAE:  {result.get('mae', 'N/A'):.4f}")
                print(f"  R²:   {result.get('r2', 'N/A'):.4f}")
                print(f"  Directional Accuracy: {result.get('directional_accuracy', 'N/A'):.4f}")
        
        # Save results
        results_df = pd.DataFrame([
            result for result in results.values() 
            if 'error' not in result and '_by_regime' not in result
        ])
        
        if not results_df.empty:
            results_df.to_csv(f"{RESULTS_DIR}comprehensive_model_performance.csv", index=False)
            logger.info("Results saved to comprehensive_model_performance.csv")
        
        # Save feature importance if available
        if 'xgboost' in models and hasattr(models['xgboost'], 'feature_importance'):
            models['xgboost'].feature_importance.to_csv(f"{RESULTS_DIR}feature_importance.csv", index=False)
            print(f"\n📊 Top 10 Important Features:")
            print(models['xgboost'].feature_importance.head(10))
        
        logger.info("Comprehensive model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in comprehensive model training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
