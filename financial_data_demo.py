#!/usr/bin/env python3
"""
Quick Start Script for Time Series Forecasting Project
This script runs a simplified version of the pipeline for testing and demonstration
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def quick_demo():
    """Run a quick demo of the time series forecasting system"""
    
    print("🚀 Time Series Forecasting - Quick Demo")
    print("=" * 50)
    
    # Step 1: Quick Data Collection
    print("\n📊 Step 1: Collecting sample data...")
    
    # Get a small sample of data for demo
    symbols = ['AAPL', 'BTC-USD', 'SPY']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    all_data = []
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty:
                data = data.reset_index()
                data['Symbol'] = symbol
                data['Market_Type'] = 'Crypto' if 'USD' in symbol else ('ETF' if symbol == 'SPY' else 'Stock')
                all_data.append(data)
                print(f"   Downloaded {symbol}: {len(data)} records")
        except Exception as e:
            print(f"  ✗ Failed to download {symbol}: {e}")
    
    if not all_data:
        print("❌ No data collected. Please check your internet connection.")
        return
    
    # Combine data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"  📈 Total records: {len(combined_data)}")
    
    # Step 2: Basic Feature Engineering
    print("\n🔧 Step 2: Creating features...")
    
    # Initialize new columns
    combined_data['Returns'] = np.nan
    combined_data['SMA_10'] = np.nan
    combined_data['Volatility'] = np.nan
    
    for symbol in combined_data['Symbol'].unique():
        mask = combined_data['Symbol'] == symbol
        symbol_data = combined_data[mask].copy()
        
        # Basic features
        returns = symbol_data['Close'].pct_change()
        sma_10 = symbol_data['Close'].rolling(10).mean()
        volatility = returns.rolling(20).std()
        
        # Update in main dataframe using the original indices
        combined_data.loc[mask, 'Returns'] = returns
        combined_data.loc[mask, 'SMA_10'] = sma_10
        combined_data.loc[mask, 'Volatility'] = volatility
    
    print(f"  🔧 Features created: Returns, SMA_10, Volatility")
    
    # Step 3: Simple Analysis
    print("\n📊 Step 3: Basic analysis...")
    
    # Market statistics - only use columns that exist
    available_cols = combined_data.columns.tolist()
    stats_cols = {}
    
    if 'Returns' in available_cols:
        stats_cols['Returns'] = ['mean', 'std']
    if 'Volatility' in available_cols:
        stats_cols['Volatility'] = 'mean'
    if 'Volume' in available_cols:
        stats_cols['Volume'] = 'mean'
    
    if stats_cols:
        market_stats = combined_data.groupby('Market_Type').agg(stats_cols).round(4)
        print("  📈 Market Statistics:")
        print(market_stats)
    else:
        print("  ⚠️ No statistical columns available for analysis")
    
    # Cross-correlations
    pivot_returns = combined_data.pivot_table(
        index='Date', 
        columns='Symbol', 
        values='Returns'
    )
    
    if len(pivot_returns.columns) > 1:
        correlations = pivot_returns.corr()
        print("\n  🔗 Return Correlations:")
        print(correlations.round(3))
    
    # Step 4: Simple Prediction Demo
    print("\n🤖 Step 4: Simple prediction demo...")
    
    # Use simple linear model for demo
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare data for one symbol (AAPL)
    apple_data = combined_data[combined_data['Symbol'] == 'AAPL'].dropna()
    
    if len(apple_data) > 50:
        # Features: previous returns, SMA ratio, volatility
        apple_data['SMA_Ratio'] = apple_data['Close'] / apple_data['SMA_10']
        apple_data['Target'] = apple_data['Returns'].shift(-1)  # Next day return
        
        # Prepare training data
        features = ['Returns', 'SMA_Ratio', 'Volatility']
        apple_clean = apple_data[features + ['Target']].dropna()
        
        if len(apple_clean) > 20:
            # Split data
            split_idx = int(len(apple_clean) * 0.8)
            
            X_train = apple_clean[features].iloc[:split_idx]
            y_train = apple_clean['Target'].iloc[:split_idx]
            X_test = apple_clean[features].iloc[split_idx:]
            y_test = apple_clean['Target'].iloc[split_idx:]
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            print(f"  🎯 Simple Model Results (AAPL):")
            print(f"    • Training samples: {len(X_train)}")
            print(f"    • Test samples: {len(X_test)}")
            print(f"    • RMSE: {np.sqrt(mse):.4f}")
            print(f"    • R²: {r2:.4f}")
            
            # Feature importance (coefficients)
            print(f"  📊 Feature Coefficients:")
            for feature, coef in zip(features, model.coef_):
                print(f"    • {feature}: {coef:.4f}")
        else:
            print("  ⚠️ Insufficient clean data for modeling")
    else:
        print("  ⚠️ Insufficient AAPL data for modeling")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ QUICK DEMO COMPLETED!")
    print("=" * 50)
    print("\n🎉 What you've seen:")
    print("  • Multi-market data collection (Stocks, Crypto, ETFs)")
    print("  • Basic feature engineering (returns, moving averages, volatility)")
    print("  • Cross-market correlation analysis")
    print("  • Simple machine learning prediction")
    print("\n🚀 Ready for the full pipeline?")
    print("  Run: python main_pipeline.py")
    print("\n📚 Or explore in Jupyter:")
    print("  Open: SageMaker_Demo.ipynb")

if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please ensure you have internet connection and required packages installed.")
        print("Run: pip install -r requirements.txt") 