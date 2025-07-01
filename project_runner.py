#!/usr/bin/env python3
"""
Time Series Forecasting Project - Main Runner
This script runs the complete project workflow step by step
"""

import subprocess
import sys
import os
from datetime import datetime

# Import project configuration
from config import configure_aws_environment

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"🚀 {title}")
    print("="*70)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def run_script(script_name, description):
    """Run a Python script and return success status"""
    print_header(f"STEP: {description}")
    print_info(f"Running: {script_name}")
    
    try:
        # Run without capturing output so we can see real-time results
        result = subprocess.run([sys.executable, script_name], 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print_success(f"{description} completed successfully!")
            return True
        else:
            print_error(f"{description} failed with exit code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"{description} timed out after 5 minutes")
        return False
    except Exception as e:
        print_error(f"Error running {script_name}: {str(e)}")
        return False

def check_aws_credentials():
    """Check and configure AWS credentials from config.py"""
    try:
        # Configure AWS environment from config file
        if configure_aws_environment():
            print_success("AWS credentials configured from config.py")
            return True
        else:
            print_error("Failed to configure AWS credentials from config.py")
            return False
    except Exception as e:
        print_error(f"Error configuring AWS credentials: {str(e)}")
        print_info("Please check your AWS credentials in config.py")
        return False

def main():
    """Main function to run the complete project workflow"""
    print("🎯 Time Series Forecasting Project - Complete Workflow")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 0: Check AWS Configuration
    print_header("STEP 0: AWS CONFIGURATION CHECK")
    if not check_aws_credentials():
        print_error("AWS credentials not configured properly")
        print_info("Please update your AWS credentials in config.py and try again")
        return False
    
    if not run_script("aws_configuration_test.py", "AWS Configuration Test"):
        print_error("AWS configuration test failed!")
        print_info("Please fix AWS configuration and try again")
        return False
    
    # Project workflow steps
    steps = [
        ("financial_data_demo.py", "Quick Demo (Financial Data Analysis)"),
        ("data_collector.py", "Data Collection from Multiple Markets"),
        ("feature_engineer.py", "Feature Engineering & Technical Indicators"),
        ("regime_detector.py", "Market Regime Detection"),
        ("ml_models.py", "Machine Learning Model Training"),
        ("main_pipeline.py", "Complete Pipeline Execution")
    ]
    
    successful_steps = 0
    total_steps = len(steps)
    
    for step_file, step_description in steps:
        if os.path.exists(step_file):
            if run_script(step_file, step_description):
                successful_steps += 1
            else:
                print_error(f"Failed at: {step_description}")
                user_choice = input("\nDo you want to continue with next steps? (y/n): ").lower()
                if user_choice != 'y':
                    break
        else:
            print_error(f"Script not found: {step_file}")
    
    # Final summary
    print_header("PROJECT EXECUTION SUMMARY")
    print_info(f"Completed steps: {successful_steps}/{total_steps}")
    
    if successful_steps == total_steps:
        print_success("🎉 All steps completed successfully!")
        print_success("Your Time Series Forecasting project is complete!")
    elif successful_steps > 0:
        print_info(f"✨ Partially completed: {successful_steps} out of {total_steps} steps")
        print_info("You can run individual scripts to retry failed steps")
    else:
        print_error("❌ No steps completed successfully")
        print_info("Please check error messages and fix issues")
    
    print_info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return successful_steps > 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Project execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1) 