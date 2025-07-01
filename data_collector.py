# Data Collection Module for Time Series Forecasting Project
import pandas as pd
import numpy as np
import yfinance as yf
import boto3
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class FinancialDataCollector:
    """
    Comprehensive financial data collector for multiple markets
    """
    
    def __init__(self, s3_bucket=None):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3', region_name=AWS_REGION) if s3_bucket else None
        
    def download_symbol_data(self, symbol, start_date, end_date, market_type):
        """Download data for a single symbol"""
        try:
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
            
            # Download data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Reset index and standardize columns
            data = data.reset_index()
            
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten multi-level columns
                data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
            
            data['Symbol'] = symbol.replace('-USD', '') if market_type == 'Crypto' else symbol
            data['Market_Type'] = market_type
            
            # Standardize column names - handle different possible column names
            column_mapping = {
                'Date': 'date',
                'Open': 'open', 
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'close',  # Use adjusted close if available
                'Volume': 'volume'
            }
            data = data.rename(columns=column_mapping)
            
            # Ensure we have all required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns for {symbol}: {missing_columns}")
                return None
            
            # Select only needed columns
            columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'Symbol', 'Market_Type']
            data = data[columns_to_keep]
            
            logger.info(f"✓ Downloaded {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"✗ Error downloading {symbol}: {str(e)}")
            return None
    
    def collect_market_data(self, symbols, market_type, start_date=START_DATE, end_date=END_DATE, max_workers=1):
        """Collect data for multiple symbols using parallel processing"""
        logger.info(f"Collecting {market_type} data for {len(symbols)} symbols...")
        
        all_data = []
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(self.download_symbol_data, symbol, start_date, end_date, market_type): symbol
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None:
                        all_data.append(data)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
        
        if all_data:
            # Standardize all datasets before concatenation
            standardized_data = []
            expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'Symbol', 'Market_Type']
            
            for data in all_data:
                # Handle multi-level columns if they exist
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
                
                # Ensure we have the expected columns
                if all(col in data.columns for col in expected_columns):
                    data_clean = data[expected_columns].copy()
                    standardized_data.append(data_clean)
                else:
                    missing_cols = [col for col in expected_columns if col not in data.columns]
                    logger.warning(f"Skipping dataset missing columns: {missing_cols}")
            
            if standardized_data:
                combined_data = pd.concat(standardized_data, ignore_index=True)
                logger.info(f"✓ {market_type} data collection complete: {len(combined_data)} total records")
                return combined_data
            else:
                logger.warning(f"No valid data collected for {market_type}")
                return pd.DataFrame()
        else:
            logger.warning(f"No data collected for {market_type}")
            return pd.DataFrame()
    
    def collect_all_markets(self):
        """Collect data from all financial markets"""
        logger.info("Starting comprehensive data collection...")
        
        # Collect data from all markets
        stock_data = self.collect_market_data(SP500_SYMBOLS, 'Stock')
        crypto_data = self.collect_market_data(CRYPTO_SYMBOLS, 'Crypto')
        etf_data = self.collect_market_data(ETF_SYMBOLS, 'ETF')
        
        # Combine all data
        all_datasets = [df for df in [stock_data, crypto_data, etf_data] if not df.empty]
        
        if all_datasets:
            # Ensure all datasets have the same column structure before concatenation
            standardized_datasets = []
            expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'Symbol', 'Market_Type']
            
            for dataset in all_datasets:
                # Handle multi-level columns if they exist
                if isinstance(dataset.columns, pd.MultiIndex):
                    dataset.columns = [col[0] if col[1] == '' else col[0] for col in dataset.columns]
                
                # Ensure only expected columns are kept
                available_columns = [col for col in expected_columns if col in dataset.columns]
                if len(available_columns) == len(expected_columns):
                    dataset_clean = dataset[expected_columns].copy()
                    standardized_datasets.append(dataset_clean)
                else:
                    logger.warning(f"Dataset missing required columns: {set(expected_columns) - set(available_columns)}")
            
            if standardized_datasets:
                combined_data = pd.concat(standardized_datasets, ignore_index=True)
                logger.info(f"Combined datasets successfully: {len(combined_data)} total records")
                
                # Clean and sort data
                combined_data = self.clean_data(combined_data)
            else:
                logger.error("No datasets could be standardized for concatenation")
                return pd.DataFrame()
            
            logger.info(f"✓ All markets data collected: {len(combined_data)} total records")
            logger.info(f"✓ Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
            logger.info(f"✓ Market distribution: {combined_data['Market_Type'].value_counts().to_dict()}")
            
            return combined_data
        else:
            logger.error("No data collected from any market")
            return pd.DataFrame()
    
    def clean_data(self, df):
        """Clean and standardize the collected data"""
        logger.info("Cleaning and standardizing data...")
        
        # Debug: Check what columns we actually have
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Check which essential columns exist
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'Symbol', 'Market_Type']
        available_columns = [col for col in required_columns if col in df.columns]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.warning("Cannot proceed with data cleaning due to missing columns")
            return pd.DataFrame()
        
        # Remove rows with missing values only in available essential columns
        initial_rows = len(df)
        df = df.dropna(subset=available_columns)
        logger.info(f"Removed {initial_rows - len(df)} rows with missing essential values")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['date', 'Symbol'])
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Remove rows with zero or negative prices
        initial_rows = len(df)
        df = df[(df['close'] > 0) & (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0)]
        logger.info(f"Removed {initial_rows - len(df)} rows with invalid prices")
        
        # Sort by date and symbol
        df = df.sort_values(['date', 'Symbol']).reset_index(drop=True)
        
        return df
    
    def save_to_s3(self, data, key):
        """Save data to S3 bucket"""
        if self.s3_client and self.s3_bucket:
            try:
                csv_buffer = data.to_csv(index=False)
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=key,
                    Body=csv_buffer
                )
                logger.info(f"✓ Data saved to s3://{self.s3_bucket}/{key}")
                return True
            except Exception as e:
                logger.error(f"Error saving to S3: {str(e)}")
                return False
        return False
    
    def save_to_local(self, data, filename):
        """Save data to local file"""
        try:
            filepath = f"{DATA_DIR}{filename}"
            data.to_csv(filepath, index=False)
            logger.info(f"✓ Data saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving locally: {str(e)}")
            return False

def main():
    """Main function to run data collection"""
    logger.info("Starting Financial Data Collection...")
    
    # Initialize collector
    collector = FinancialDataCollector()
    
    # Collect all market data
    financial_data = collector.collect_all_markets()
    
    if not financial_data.empty:
        # Save data locally
        collector.save_to_local(financial_data, 'raw_financial_data.csv')
        
        # Display summary statistics
        print("\n" + "="*60)
        print("DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"Total records: {len(financial_data):,}")
        print(f"Date range: {financial_data['date'].min()} to {financial_data['date'].max()}")
        print(f"Unique symbols: {financial_data['Symbol'].nunique()}")
        print("\nMarket Distribution:")
        for market, count in financial_data['Market_Type'].value_counts().items():
            print(f"  {market}: {count:,} records")
        
        print("\nSample Data:")
        print(financial_data.head())
        
        print("\nData Types:")
        print(financial_data.dtypes)
        
    else:
        logger.error("No data was collected!")

if __name__ == "__main__":
    main() 