# Configuration File for Time Series Forecasting Project
import os
from datetime import datetime

# AWS Configuration
AWS_REGION = 'us-east-1'  # AWS Academy default region
SAGEMAKER_ROLE = 'LabRole'  # Pre-configured role in AWS Academy

# AWS Credentials - Use environment variables for security
# Set these in your environment or .env file:
# export AWS_ACCESS_KEY_ID="your_key_here"
# export AWS_SECRET_ACCESS_KEY="your_secret_here" 
# export AWS_SESSION_TOKEN="your_token_here"
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN', '')

# AWS Account Information - Set via environment variable
AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID', '')

# Function to configure AWS environment
def configure_aws_environment():
    """Configure AWS environment variables from config"""
    import os
    
    # Only set environment variables if they're not already set and we have values
    if AWS_ACCESS_KEY_ID and not os.getenv('AWS_ACCESS_KEY_ID'):
        os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    if AWS_SECRET_ACCESS_KEY and not os.getenv('AWS_SECRET_ACCESS_KEY'):
        os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    if AWS_SESSION_TOKEN and not os.getenv('AWS_SESSION_TOKEN'):
        os.environ['AWS_SESSION_TOKEN'] = AWS_SESSION_TOKEN
    if not os.getenv('AWS_DEFAULT_REGION'):
        os.environ['AWS_DEFAULT_REGION'] = AWS_REGION
    
    # Check if we have the required credentials
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing AWS credentials: {', '.join(missing_vars)}")
        print("Please set these environment variables or add them to your .env file")
        return False
    
    return True

# S3 Configuration
S3_BUCKET_NAME = f'timeseries-forecasting-{datetime.now().strftime("%Y%m%d%H%M%S")}'
S3_DATA_PREFIX = 'data/'
S3_MODELS_PREFIX = 'models/'
S3_RESULTS_PREFIX = 'results/'

# Data Collection Parameters
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'
TARGET_ROWS = 50000  # Target number of rows per dataset

# Stock symbols for data collection
SP500_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
    'V', 'PG', 'JPM', 'HD', 'MA', 'PFE', 'BAC', 'ABBV', 'KO', 'AVGO',
    'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'MRK', 'VZ', 'ADBE', 'WMT', 'CRM'
]

CRYPTO_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
    'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LTC-USD', 'UNI-USD',
    'LINK-USD', 'ALGO-USD', 'BCH-USD'
]

ETF_SYMBOLS = [
    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND',
    'AGG', 'GLD', 'SLV', 'USO', 'XLF', 'XLE', 'XLK', 'XLV'
]

# Model Parameters
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Technical Indicators Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# Regime Detection Parameters
VOLATILITY_CLUSTERS = 3
CORRELATION_WINDOW = 20
TREND_WINDOW = 20

# Model Training Parameters
ARIMA_ORDER = (1, 1, 1)
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}

LSTM_PARAMS = {
    'units': 50,
    'epochs': 50,
    'batch_size': 32,
    'sequence_length': 60
}

# File Paths
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
RESULTS_DIR = 'results/'
LOGS_DIR = 'logs/'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO' 