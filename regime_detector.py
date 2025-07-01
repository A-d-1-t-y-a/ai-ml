# Market Regime Detection Module for Time Series Forecasting Project
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import logging

# Try to import HDBSCAN for alternative clustering
try:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: HDBSCAN not available, using K-means only")

from config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Advanced market regime detection system for financial markets
    """
    
    def __init__(self, clustering_method='kmeans'):
        self.regime_models = {}
        self.regime_scalers = {}
        self.clustering_method = clustering_method
        
    def detect_volatility_regimes(self, df):
        """Detect volatility regimes using K-means clustering"""
        logger.info("Detecting volatility regimes...")
        
        regime_data = []
        
        for market_type in df['Market_Type'].unique():
            market_data = df[df['Market_Type'] == market_type].copy()
            
            # Calculate daily market-level volatility
            daily_data = market_data.groupby('date').agg({
                'returns': 'std',
                'volatility_20': 'mean',
                'volume': 'sum'
            }).reset_index()
            
            if len(daily_data) < 30:  # Need sufficient data for clustering
                logger.warning(f"Insufficient data for {market_type} volatility regime detection")
                continue
            
            # Prepare features for clustering
            features = daily_data[['returns', 'volatility_20']].fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=VOLATILITY_CLUSTERS, random_state=42, n_init=10)
            vol_regimes = kmeans.fit_predict(features_scaled)
            
            # Assign regime labels based on cluster centers
            cluster_centers = kmeans.cluster_centers_
            
            # Sort clusters by average volatility level
            center_vol_levels = [np.mean(center) for center in cluster_centers]
            sorted_indices = np.argsort(center_vol_levels)
            
            regime_mapping = {
                sorted_indices[0]: 'Low_Vol',
                sorted_indices[1]: 'Medium_Vol',
                sorted_indices[2]: 'High_Vol'
            }
            
            vol_regime_labels = [regime_mapping[regime] for regime in vol_regimes]
            
            daily_data['Market_Type'] = market_type
            daily_data['volatility_regime'] = vol_regime_labels
            regime_data.append(daily_data)
            
            # Store model for future predictions
            self.regime_models[f'{market_type}_volatility'] = kmeans
            self.regime_scalers[f'{market_type}_volatility'] = scaler
            
            regime_counts = pd.Series(vol_regime_labels).value_counts().to_dict()
            logger.info(f"{market_type} volatility regimes: {regime_counts}")
        
        if regime_data:
            volatility_regimes = pd.concat(regime_data, ignore_index=True)
            return volatility_regimes
        else:
            return pd.DataFrame()
    
    def detect_volatility_regimes_hdbscan(self, df):
        """Detect volatility regimes using HDBSCAN clustering (alternative method)"""
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available, falling back to K-means")
            return self.detect_volatility_regimes(df)
            
        logger.info("Detecting volatility regimes using HDBSCAN...")
        
        regime_data = []
        
        for market_type in df['Market_Type'].unique():
            market_data = df[df['Market_Type'] == market_type].copy()
            
            # Calculate daily market-level volatility
            daily_data = market_data.groupby('date').agg({
                'returns': 'std',
                'volatility_20': 'mean',
                'volume': 'sum'
            }).reset_index()
            
            if len(daily_data) < 30:
                logger.warning(f"Insufficient data for {market_type} volatility regime detection")
                continue
            
            # Prepare features for clustering
            features = daily_data[['returns', 'volatility_20']].fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply HDBSCAN clustering
            clusterer = HDBSCAN(min_cluster_size=10, min_samples=5)
            cluster_labels = clusterer.fit_predict(features_scaled)
            
            # Handle noise points and map to volatility levels
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
            
            if n_clusters < 2:
                logger.warning(f"HDBSCAN found insufficient clusters for {market_type}, using K-means fallback")
                # Fallback to K-means
                kmeans = KMeans(n_clusters=VOLATILITY_CLUSTERS, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_scaled)
                
            # Map clusters to volatility regime labels
            vol_regime_labels = []
            for label in cluster_labels:
                if label == -1:  # Noise points
                    vol_regime_labels.append('Medium_Vol')
                else:
                    # Map based on cluster center volatility
                    cluster_mask = cluster_labels == label
                    avg_vol = features_scaled[cluster_mask, 1].mean()  # volatility_20 feature
                    
                    if avg_vol > 0.5:
                        vol_regime_labels.append('High_Vol')
                    elif avg_vol < -0.5:
                        vol_regime_labels.append('Low_Vol')
                    else:
                        vol_regime_labels.append('Medium_Vol')
            
            daily_data['Market_Type'] = market_type
            daily_data['volatility_regime'] = vol_regime_labels
            regime_data.append(daily_data)
            
            regime_counts = pd.Series(vol_regime_labels).value_counts().to_dict()
            logger.info(f"{market_type} volatility regimes (HDBSCAN): {regime_counts}")
        
        if regime_data:
            volatility_regimes = pd.concat(regime_data, ignore_index=True)
            return volatility_regimes
        else:
            return pd.DataFrame()
    
    def detect_trend_regimes(self, df):
        """Detect trend regimes using moving averages and price momentum"""
        logger.info("Detecting trend regimes...")
        
        trend_data = []
        
        for market_type in df['Market_Type'].unique():
            market_data = df[df['Market_Type'] == market_type].copy()
            
            # Calculate market-level daily data
            daily_data = market_data.groupby('date').agg({
                'close': 'mean',
                'returns': 'mean',
                'sma_20': 'mean',
                'sma_50': 'mean'
            }).reset_index()
            
            if len(daily_data) < 50:
                logger.warning(f"Insufficient data for {market_type} trend regime detection")
                continue
            
            # Calculate trend indicators
            daily_data['price_sma20_ratio'] = daily_data['close'] / daily_data['sma_20']
            daily_data['price_sma50_ratio'] = daily_data['close'] / daily_data['sma_50']
            daily_data['sma20_sma50_ratio'] = daily_data['sma_20'] / daily_data['sma_50']
            
            # Rolling correlation for mean reversion detection
            daily_data['returns_lag1'] = daily_data['returns'].shift(1)
            daily_data['mean_reversion_signal'] = daily_data['returns'].rolling(TREND_WINDOW).corr(daily_data['returns_lag1'])
            
            # Define trend regimes based on multiple conditions
            conditions = [
                (daily_data['price_sma20_ratio'] > 1.02) & (daily_data['sma20_sma50_ratio'] > 1.01),
                (daily_data['price_sma20_ratio'] < 0.98) & (daily_data['sma20_sma50_ratio'] < 0.99),
            ]
            
            choices = ['Uptrend', 'Downtrend']
            
            daily_data['trend_regime'] = np.select(conditions, choices, default='Sideways')
            
            daily_data['Market_Type'] = market_type
            trend_data.append(daily_data)
            
            trend_counts = daily_data['trend_regime'].value_counts().to_dict()
            logger.info(f"{market_type} trend regimes: {trend_counts}")
        
        if trend_data:
            trend_regimes = pd.concat(trend_data, ignore_index=True)
            return trend_regimes
        else:
            return pd.DataFrame()
    
    def detect_correlation_regimes(self, df):
        """Detect cross-market correlation regimes"""
        logger.info("Detecting correlation regimes...")
        
        # Calculate daily market returns for correlation analysis
        market_returns = df.groupby(['date', 'Market_Type'])['returns'].mean().unstack(fill_value=0)
        
        if market_returns.empty or len(market_returns.columns) < 2:
            logger.warning("Insufficient market data for correlation regime detection")
            return pd.DataFrame()
        
        correlation_data = []
        window = CORRELATION_WINDOW
        
        for i in range(window, len(market_returns)):
            date = market_returns.index[i]
            window_data = market_returns.iloc[i-window:i]
            
            # Calculate correlations between all market pairs
            correlations = []
            market_types = window_data.columns.tolist()
            
            for j, market1 in enumerate(market_types):
                for market2 in market_types[j+1:]:
                    corr_value = window_data[market1].corr(window_data[market2])
                    if not pd.isna(corr_value):
                        correlations.append(abs(corr_value))
            
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Define correlation regime based on average absolute correlation
            if avg_correlation > 0.7:
                corr_regime = 'High_Correlation'
            elif avg_correlation < 0.3:
                corr_regime = 'Low_Correlation'
            else:
                corr_regime = 'Medium_Correlation'
            
            correlation_data.append({
                'date': date,
                'avg_correlation': avg_correlation,
                'correlation_regime': corr_regime
            })
        
        if correlation_data:
            correlation_regimes = pd.DataFrame(correlation_data)
            corr_counts = correlation_regimes['correlation_regime'].value_counts().to_dict()
            logger.info(f"Cross-market correlation regimes: {corr_counts}")
            return correlation_regimes
        else:
            return pd.DataFrame()
    
    def combine_regime_signals(self, volatility_regimes, trend_regimes, correlation_regimes):
        """Combine all regime signals into unified regime classification"""
        logger.info("Combining regime signals...")
        
        if volatility_regimes.empty or trend_regimes.empty:
            logger.warning("Missing regime data for combination")
            return pd.DataFrame()
        
        # Merge volatility and trend regimes
        combined_regimes = volatility_regimes.merge(
            trend_regimes[['date', 'Market_Type', 'trend_regime']], 
            on=['date', 'Market_Type'], 
            how='left'
        )
        
        # Merge correlation regimes if available
        if not correlation_regimes.empty:
            combined_regimes = combined_regimes.merge(
                correlation_regimes[['date', 'correlation_regime']], 
                on='date', 
                how='left'
            )
        else:
            combined_regimes['correlation_regime'] = 'Unknown'
        
        # Create composite regime signal
        def create_composite_regime(row):
            vol = row['volatility_regime']
            trend = row['trend_regime'] 
            corr = row.get('correlation_regime', 'Unknown')
            
            # Define composite regimes based on combinations
            if vol == 'High_Vol' and corr == 'High_Correlation':
                return 'Crisis_Regime'
            elif vol == 'Low_Vol' and trend == 'Uptrend':
                return 'Bull_Market'
            elif vol == 'Low_Vol' and trend == 'Downtrend':
                return 'Bear_Market'
            elif vol == 'Medium_Vol' and trend == 'Sideways':
                return 'Consolidation'
            elif vol == 'High_Vol' and corr == 'Low_Correlation':
                return 'Uncertainty_Regime'
            else:
                return 'Transition_Regime'
        
        combined_regimes['composite_regime'] = combined_regimes.apply(create_composite_regime, axis=1)
        
        regime_distribution = combined_regimes['composite_regime'].value_counts().to_dict()
        logger.info(f"Composite regime distribution: {regime_distribution}")
        
        return combined_regimes
    
    def create_regime_features(self, df, regime_df):
        """Create regime-based features for modeling"""
        logger.info("Creating regime-based features...")
        
        # Merge regime information with main dataset
        df_with_regimes = df.merge(
            regime_df[['date', 'Market_Type', 'volatility_regime', 'trend_regime', 'composite_regime']], 
            on=['date', 'Market_Type'], 
            how='left'
        )
        
        # Create regime transition features
        regime_features = []
        
        for symbol in df_with_regimes['Symbol'].unique():
            symbol_data = df_with_regimes[df_with_regimes['Symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
            
            # Previous regime information
            symbol_data['prev_volatility_regime'] = symbol_data['volatility_regime'].shift(1)
            symbol_data['prev_trend_regime'] = symbol_data['trend_regime'].shift(1)
            symbol_data['prev_composite_regime'] = symbol_data['composite_regime'].shift(1)
            
            # Regime change indicators
            symbol_data['volatility_regime_change'] = (
                symbol_data['volatility_regime'] != symbol_data['prev_volatility_regime']
            ).astype(int)
            
            symbol_data['trend_regime_change'] = (
                symbol_data['trend_regime'] != symbol_data['prev_trend_regime']
            ).astype(int)
            
            symbol_data['composite_regime_change'] = (
                symbol_data['composite_regime'] != symbol_data['prev_composite_regime']
            ).astype(int)
            
            # Regime duration (how long has current regime lasted)
            symbol_data['regime_duration'] = 0
            current_regime = None
            duration = 0
            
            for idx, row in symbol_data.iterrows():
                if row['composite_regime'] == current_regime:
                    duration += 1
                else:
                    current_regime = row['composite_regime']
                    duration = 1
                symbol_data.at[idx, 'regime_duration'] = duration
            
            regime_features.append(symbol_data)
        
        if regime_features:
            final_df = pd.concat(regime_features, ignore_index=True)
            logger.info(f"Regime features created for {len(final_df['Symbol'].unique())} symbols")
            return final_df
        else:
            return df_with_regimes
    
    def detect_all_regimes(self, df):
        """Main function to detect all types of market regimes"""
        logger.info("Starting comprehensive regime detection...")
        
        # Detect different types of regimes
        volatility_regimes = self.detect_volatility_regimes(df)
        trend_regimes = self.detect_trend_regimes(df)
        correlation_regimes = self.detect_correlation_regimes(df)
        
        # Combine regime signals
        if not volatility_regimes.empty and not trend_regimes.empty:
            combined_regimes = self.combine_regime_signals(
                volatility_regimes, trend_regimes, correlation_regimes
            )
            
            # Create regime-based features
            df_with_regimes = self.create_regime_features(df, combined_regimes)
            
            logger.info("Regime detection completed successfully")
            return df_with_regimes, combined_regimes
        else:
            logger.error("Failed to detect regimes")
            return df, pd.DataFrame()
    
    def predict_regime(self, new_data, market_type):
        """Predict regime for new data using trained models"""
        if f'{market_type}_volatility' not in self.regime_models:
            logger.warning(f"No trained model for {market_type}")
            return 'Unknown'
        
        try:
            model = self.regime_models[f'{market_type}_volatility']
            scaler = self.regime_scalers[f'{market_type}_volatility']
            
            features = new_data[['returns', 'volatility_20']].fillna(0)
            features_scaled = scaler.transform(features.values.reshape(1, -1))
            
            cluster = model.predict(features_scaled)[0]
            
            # Map cluster to regime label (simplified)
            regime_mapping = {0: 'Low_Vol', 1: 'Medium_Vol', 2: 'High_Vol'}
            return regime_mapping.get(cluster, 'Unknown')
            
        except Exception as e:
            logger.error(f"Error predicting regime: {str(e)}")
            return 'Unknown'

def main():
    """Main function to test regime detection"""
    logger.info("Testing Market Regime Detection...")
    
    try:
        # Load processed data
        data = pd.read_csv(f"{DATA_DIR}enhanced_features.csv")
        data['date'] = pd.to_datetime(data['date'])
        
        # Take a sample for testing
        sample_data = data.head(5000)
        
        # Initialize regime detector
        detector = MarketRegimeDetector()
        
        # Detect regimes
        data_with_regimes, regime_summary = detector.detect_all_regimes(sample_data)
        
        # Save results
        data_with_regimes.to_csv(f"{DATA_DIR}data_with_regimes.csv", index=False)
        
        if not regime_summary.empty:
            regime_summary.to_csv(f"{DATA_DIR}regime_summary.csv", index=False)
        
        print("\nREGIME DETECTION SUMMARY")
        print("="*50)
        print(f"Input records: {len(sample_data)}")
        print(f"Output records: {len(data_with_regimes)}")
        
        if 'composite_regime' in data_with_regimes.columns:
            print("\nComposite Regime Distribution:")
            for regime, count in data_with_regimes['composite_regime'].value_counts().items():
                print(f"  {regime}: {count} ({count/len(data_with_regimes)*100:.1f}%)")
        
        print(f"\nRegime features added: {len(data_with_regimes.columns) - len(sample_data.columns)}")
        
    except FileNotFoundError:
        logger.error("Enhanced features file not found. Please run feature_engineering.py first.")
    except Exception as e:
        logger.error(f"Error in regime detection: {str(e)}")

if __name__ == "__main__":
    main() 