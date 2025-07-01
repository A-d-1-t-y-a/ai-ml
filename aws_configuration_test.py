#!/usr/bin/env python3
"""
AWS Configuration Test Script
This script tests all AWS-related configurations for the Time Series Forecasting Project
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project configuration
from config import configure_aws_environment, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_success(message):
    """Print success message"""
    print(f" {message}")

def print_warning(message):
    """Print warning message"""
    print(f"  {message}")

def print_error(message):
    """Print error message"""
    print(f" {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ  {message}")

def test_aws_credentials():
    """Test if AWS credentials are configured"""
    print_header("AWS CREDENTIALS TEST")
    
    try:
        # Configure AWS environment from config file
        configure_aws_environment()
        print_success("AWS credentials loaded from config.py")
        
        # Verify credentials in config
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            print_error("AWS credentials not configured in config.py!")
            print_info("Please update the following in config.py:")
            print_info("  - AWS_ACCESS_KEY_ID")
            print_info("  - AWS_SECRET_ACCESS_KEY")
            print_info("  - AWS_SESSION_TOKEN (for AWS Academy)")
            return False
        
        print_success("AWS credentials found in configuration")
        print_info(f"Access Key: {AWS_ACCESS_KEY_ID[:10]}...")
        
        # Check if session token exists (for temporary credentials)
        if AWS_SESSION_TOKEN:
            print_info("Using temporary credentials (AWS Academy)")
        else:
            print_info("Using permanent credentials")
        
        # Try to create a session to verify credentials work
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials and credentials.access_key:
            print_success("AWS session created successfully")
            return True
        else:
            print_error("Failed to create AWS session with configured credentials")
            return False
            
    except Exception as e:
        print_error(f"Error checking AWS credentials: {str(e)}")
        return False

def test_aws_region():
    """Test AWS region configuration"""
    print_header("AWS REGION TEST")
    
    try:
        session = boto3.Session()
        region = session.region_name
        
        if region:
            print_success(f"AWS region configured: {region}")
            return region
        else:
            # Try to get from environment or config
            region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')
            if region:
                print_success(f"AWS region from environment: {region}")
                return region
            else:
                print_warning("No AWS region configured")
                print_info("Consider setting AWS_DEFAULT_REGION environment variable")
                print_info("or configuring region in ~/.aws/config")
                return 'us-east-1'  # Default region
                
    except Exception as e:
        print_error(f"Error checking AWS region: {str(e)}")
        return 'us-east-1'

def test_sts_identity():
    """Test STS (Security Token Service) to get current identity"""
    print_header("AWS IDENTITY TEST")
    
    try:
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        
        print_success("Successfully authenticated with AWS")
        print_info(f"Account ID: {identity.get('Account', 'Unknown')}")
        print_info(f"User ARN: {identity.get('Arn', 'Unknown')}")
        print_info(f"User ID: {identity.get('UserId', 'Unknown')}")
        
        return True
        
    except NoCredentialsError:
        print_error("No AWS credentials configured")
        return False
    except PartialCredentialsError:
        print_error("Incomplete AWS credentials")
        return False
    except ClientError as e:
        print_error(f"AWS authentication failed: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print_error(f"Unexpected error during authentication: {str(e)}")
        return False

def test_s3_access(bucket_name=None):
    """Test S3 access"""
    print_header("S3 ACCESS TEST")
    
    try:
        s3_client = boto3.client('s3')
        
        # List available buckets
        print_info("Testing S3 connectivity...")
        response = s3_client.list_buckets()
        
        print_success("S3 connection successful")
        buckets = response.get('Buckets', [])
        
        if buckets:
            print_info(f"Found {len(buckets)} S3 buckets:")
            for bucket in buckets[:5]:  # Show first 5 buckets
                print_info(f"  - {bucket['Name']}")
            if len(buckets) > 5:
                print_info(f"  ... and {len(buckets) - 5} more")
        else:
            print_warning("No S3 buckets found in your account")
        
        return True
        
    except NoCredentialsError:
        print_error("No AWS credentials for S3 access")
        return False
    except ClientError as e:
        print_error(f"S3 access failed: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print_error(f"Unexpected S3 error: {str(e)}")
        return False

def test_environment_variables():
    """Test relevant environment variables"""
    print_header("ENVIRONMENT VARIABLES TEST")
    
    aws_vars = {
        'AWS_ACCESS_KEY_ID': 'AWS Access Key ID',
        'AWS_SECRET_ACCESS_KEY': 'AWS Secret Access Key',
        'AWS_DEFAULT_REGION': 'Default AWS Region',
        'AWS_REGION': 'AWS Region',
        'AWS_PROFILE': 'AWS Profile'
    }
    
    found_vars = {}
    for var, description in aws_vars.items():
        value = os.environ.get(var)
        if value:
            if 'KEY' in var:
                print_success(f"{description}: {'*' * 10}...")
            else:
                print_success(f"{description}: {value}")
            found_vars[var] = value
        else:
            print_info(f"{description}: Not set")
    
    if found_vars:
        print_info(f"Found {len(found_vars)} AWS environment variables")
    else:
        print_info("No AWS environment variables found (this is OK if using other methods)")
    
    return found_vars

def main():
    """Main function to run all AWS configuration tests"""
    print(" AWS Configuration Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Run all tests
    test_results['credentials'] = test_aws_credentials()
    test_results['region'] = test_aws_region()
    test_results['identity'] = test_sts_identity()
    test_results['s3'] = test_s3_access()
    test_results['env_vars'] = test_environment_variables()
    
    # Final summary
    print_header("TEST SUMMARY")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    if passed_tests >= 3:
        print_success(f"{passed_tests}/{total_tests} tests passed! ")
        print_success("Your AWS configuration looks good!")
    else:
        print_error(f"Only {passed_tests}/{total_tests} tests passed")
        print_error("AWS configuration needs attention")
    
    print_info(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed_tests >= 3

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error during testing: {str(e)}")
        sys.exit(1)
