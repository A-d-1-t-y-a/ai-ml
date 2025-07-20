# Feature Engineering Module for Time Series Forecasting Project
import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("Warning: pandas_ta not available, some features may be limited")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from scipy.stats import zscore
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for financial time series data
    """
    
    def __init__(self):
        self.scalers = {}
        
    def create_technical_indicators(self, df):
        """Create comprehensive technical analysis indicators"""
        logger.info("Creating technical indicators...")
        
        technical_data = []
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            if len(symbol_data) < 50:  # Skip symbols with insufficient data
                logger.warning(f"Skipping {symbol}: insufficient data ({len(symbol_data)} records)")
                continue
            
            # Basic price features
            symbol_data['returns'] = symbol_data['close'].pct_change()
            symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
            symbol_data['price_change'] = symbol_data['close'] - symbol_data['open']
            symbol_data['price_range'] = symbol_data['high'] - symbol_data['low']
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                symbol_data[f'sma_{window}'] = symbol_data['close'].rolling(window=window).mean()
                symbol_data[f'ema_{window}'] = symbol_data['close'].ewm(span=window).mean()
                symbol_data[f'price_sma{window}_ratio'] = symbol_data['close'] / symbol_data[f'sma_{window}']
            
            # MACD
            symbol_data['ema_12'] = symbol_data['close'].ewm(span=12).mean()
            symbol_data['ema_26'] = symbol_data['close'].ewm(span=26).mean()
            symbol_data['macd'] = symbol_data['ema_12'] - symbol_data['ema_26']
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
            symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']
            
            # RSI
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
            rs = gain / loss
            symbol_data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            symbol_data['bb_middle'] = symbol_data['close'].rolling(window=BB_PERIOD).mean()
            bb_std = symbol_data['close'].rolling(window=BB_PERIOD).std()
            symbol_data['bb_upper'] = symbol_data['bb_middle'] + (BB_STD * bb_std)
            symbol_data['bb_lower'] = symbol_data['bb_middle'] - (BB_STD * bb_std)
            symbol_data['bb_width'] = symbol_data['bb_upper'] - symbol_data['bb_lower']
            symbol_data['bb_position'] = (symbol_data['close'] - symbol_data['bb_lower']) / symbol_data['bb_width']
            
            # Volatility indicators
            for window in [5, 10, 20]:
                symbol_data[f'volatility_{window}'] = symbol_data['returns'].rolling(window=window).std()
            
            # Volume indicators
            symbol_data['volume_sma_10'] = symbol_data['volume'].rolling(window=10).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma_10']
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                symbol_data[f'close_lag_{lag}'] = symbol_data['close'].shift(lag)
                symbol_data[f'returns_lag_{lag}'] = symbol_data['returns'].shift(lag)
            
            technical_data.append(symbol_data)
        
        if technical_data:
            result_df = pd.concat(technical_data, ignore_index=True)
            logger.info(f" Technical indicators created for {len(result_df['Symbol'].unique())} symbols")
            return result_df
        else:
            logger.error("No technical indicators created")
            return pd.DataFrame()
    
    def create_cross_market_features(self, df):
        """Create features that capture cross-market relationships"""
        logger.info("Creating cross-market features...")
        
        # Create market-level aggregates for each date
        market_aggs = df.groupby(['date', 'Market_Type']).agg({
            'returns': ['mean', 'std'],
            'volatility_20': 'mean',
            'volume': 'sum',
            'rsi': 'mean'
        }).reset_index()
        
        # Flatten column names
        market_aggs.columns = ['date', 'Market_Type'] + [f"{col[0]}_{col[1]}" for col in market_aggs.columns[2:]]
        
        # Pivot to get market-specific columns
        market_pivot = market_aggs.pivot(index='date', columns='Market_Type')
        market_pivot.columns = [f"{market}_{metric}" for metric, market in market_pivot.columns]
        market_pivot = market_pivot.reset_index()
        
        # Merge with original data
        df_with_cross = df.merge(market_pivot, on='date', how='left')
        
        logger.info(f" Cross-market features created. Shape: {df_with_cross.shape}")
        return df_with_cross
    
    def create_rolling_correlations(self, df, window=20):
        """Create rolling correlation features between markets"""
        logger.info(f"Creating {window}-day rolling correlations...")
        
        correlation_features = []
        
        # Get market returns for correlation calculation
        market_returns = df.groupby(['date', 'Market_Type'])['returns'].mean().unstack(fill_value=0)
        
        if market_returns.empty:
            logger.warning("No data for correlation calculation")
            return df
        
        # Calculate rolling correlations
        for i in range(window, len(market_returns)):
            date = market_returns.index[i]
            window_data = market_returns.iloc[i-window:i]
            
            feature_row = {'date': date}
            
            # Calculate correlations between all market pairs
            if len(window_data.columns) >= 2:
                corr_matrix = window_data.corr()
                
                market_types = corr_matrix.columns.tolist()
                for j, market1 in enumerate(market_types):
                    for market2 in market_types[j+1:]:
                        if not pd.isna(corr_matrix.loc[market1, market2]):
                            feature_row[f'corr_{market1}_{market2}'] = corr_matrix.loc[market1, market2]
            
            correlation_features.append(feature_row)
        
        if correlation_features:
            corr_df = pd.DataFrame(correlation_features)
            df_with_corr = df.merge(corr_df, on='date', how='left')
            logger.info(f" Rolling correlations created: {len(corr_df.columns)-1} features")
            return df_with_corr
        else:
            logger.warning("No correlation features created")
            return df
    
    def create_momentum_features(self, df):
        """Create momentum and trend features"""
        logger.info("Creating momentum features...")
        
        momentum_data = []
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            if len(symbol_data) < 20:
                continue
            
            # Price momentum features
            for window in [5, 10, 20]:
                symbol_data[f'momentum_{window}'] = symbol_data['close'].pct_change(periods=window)
                symbol_data[f'price_acceleration_{window}'] = symbol_data[f'momentum_{window}'].diff()
            
            # Trend strength
            symbol_data['trend_strength_20'] = np.abs(symbol_data['momentum_20'])
            
            # Moving average convergence/divergence
            symbol_data['ma_convergence'] = symbol_data['sma_10'] - symbol_data['sma_20']
            symbol_data['ma_convergence_rate'] = symbol_data['ma_convergence'].diff()
            
            # Volume momentum
            symbol_data['volume_momentum_5'] = symbol_data['volume'].pct_change(periods=5)
            symbol_data['volume_momentum_10'] = symbol_data['volume'].pct_change(periods=10)
            
            momentum_data.append(symbol_data)
        
        if momentum_data:
            result_df = pd.concat(momentum_data, ignore_index=True)
            logger.info(f" Momentum features created")
            return result_df
        else:
            return df
    
    def create_market_microstructure_features(self, df):
        """Create market microstructure features"""
        logger.info("Creating market microstructure features...")
        
        microstructure_data = []
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            if len(symbol_data) < 10:
                continue
            
            # Intraday features
            symbol_data['high_low_spread'] = (symbol_data['high'] - symbol_data['low']) / symbol_data['close']
            symbol_data['open_close_spread'] = (symbol_data['close'] - symbol_data['open']) / symbol_data['open']
            
            # Volume-price relationship
            symbol_data['vwap_approx'] = (symbol_data['high'] + symbol_data['low'] + symbol_data['close']) / 3
            symbol_data['close_vwap_ratio'] = symbol_data['close'] / symbol_data['vwap_approx']
            
            # Liquidity proxy
            symbol_data['turnover'] = symbol_data['volume'] * symbol_data['close']
            symbol_data['turnover_ma_10'] = symbol_data['turnover'].rolling(window=10).mean()
            symbol_data['liquidity_ratio'] = symbol_data['turnover'] / (symbol_data['turnover_ma_10'] + 1e-8)
            
            microstructure_data.append(symbol_data)
        
        if microstructure_data:
            result_df = pd.concat(microstructure_data, ignore_index=True)
            logger.info(f" Market microstructure features created")
            return result_df
        else:
            return df
    
    def normalize_features(self, df, feature_columns=None):
        """Normalize/standardize features"""
        logger.info("Normalizing features...")
        
        if feature_columns is None:
            # Auto-detect numeric features
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude date and identifier columns
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in feature_columns if col not in exclude_cols]
        
        df_normalized = df.copy()
        
        # Normalize by symbol to avoid cross-contamination
        for symbol in df['Symbol'].unique():
            symbol_mask = df['Symbol'] == symbol
            symbol_data = df.loc[symbol_mask, feature_columns]
            
            if len(symbol_data) > 0:
                # Use StandardScaler
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(symbol_data.fillna(0))
                
                # Store scaler for later use
                self.scalers[symbol] = scaler
                
                # Replace normalized data
                df_normalized.loc[symbol_mask, feature_columns] = normalized_data
        
        logger.info(f" Normalized {len(feature_columns)} features")
        return df_normalized
    
    def create_target_variables(self, df, horizons=[1, 5, 10]):
        """Create target variables for prediction"""
        logger.info(f"Creating target variables for horizons: {horizons}")
        
        target_data = []
        
        for symbol in df['Symbol'].unique():
            symbol_data = df[df['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            for horizon in horizons:
                # Price targets
                symbol_data[f'price_target_{horizon}d'] = symbol_data['close'].shift(-horizon)
                symbol_data[f'return_target_{horizon}d'] = symbol_data['close'].pct_change(periods=horizon).shift(-horizon)
                
                # Direction targets (binary classification)
                symbol_data[f'direction_target_{horizon}d'] = (symbol_data[f'return_target_{horizon}d'] > 0).astype(int)
                
                # Volatility targets
                symbol_data[f'volatility_target_{horizon}d'] = symbol_data['returns'].rolling(window=horizon).std().shift(-horizon)
            
            target_data.append(symbol_data)
        
        if target_data:
            result_df = pd.concat(target_data, ignore_index=True)
            logger.info(f" Target variables created")
            return result_df
        else:
            return df
    
    def process_all_features(self, df):
        """Process all feature engineering steps"""
        logger.info("Starting comprehensive feature engineering...")
        
        # Step 1: Technical indicators
        df_with_tech = self.create_technical_indicators(df)
        if df_with_tech.empty:
            logger.error("Failed to create technical indicators")
            return df
        
        # Step 2: Cross-market features
        df_with_cross = self.create_cross_market_features(df_with_tech)
        
        # Step 3: Rolling correlations
        df_with_corr = self.create_rolling_correlations(df_with_cross)
        
        # Step 4: Momentum features
        df_with_momentum = self.create_momentum_features(df_with_corr)
        
        # Step 5: Market microstructure features
        df_with_micro = self.create_market_microstructure_features(df_with_momentum)
        
        # Step 6: Target variables
        df_with_targets = self.create_target_variables(df_with_micro)
        
        # Step 7: Clean final dataset
        final_df = df_with_targets.dropna(thresh=len(df_with_targets.columns) * 0.5)  # Remove rows with >50% missing values
        
        logger.info(f" Feature engineering complete. Final shape: {final_df.shape}")
        logger.info(f" Features created: {len(final_df.columns)} total columns")
        
        return final_df

def main():
    """Main function to test feature engineering"""
    logger.info("Testing Feature Engineering...")
    
    # Load sample data (assuming it exists)
    try:
        sample_data = pd.read_csv(f"{DATA_DIR}raw_financial_data.csv")
        sample_data['date'] = pd.to_datetime(sample_data['date'])
        
        # Take a smaller sample for testing
        sample_data = sample_data.head(1000)
        
        # Initialize feature engineer
        fe = FeatureEngineer()
        
        # Process features
        enhanced_data = fe.process_all_features(sample_data)
        
        # Save result
        enhanced_data.to_csv(f"{DATA_DIR}enhanced_features_sample.csv", index=False)
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        print(f"Input shape: {sample_data.shape}")
        print(f"Output shape: {enhanced_data.shape}")
        print(f"Features created: {enhanced_data.shape[1] - sample_data.shape[1]}")
        
        print("\nNew feature columns:")
        new_cols = [col for col in enhanced_data.columns if col not in sample_data.columns]
        for i, col in enumerate(new_cols[:20]):  # Show first 20
            print(f"  {i+1:2d}. {col}")
        if len(new_cols) > 20:
            print(f"  ... and {len(new_cols)-20} more features")
        
    except FileNotFoundError:
        logger.error("Raw data file not found. Please run data_collector.py first.")
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")

if __name__ == "__main__":
    main() 