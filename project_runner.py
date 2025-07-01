#!/usr/bin/env python3
"""
Time Series Forecasting Project - Main Runner
This script runs the complete project workflow step by step
"""

import subprocess
import sys
import os
from datetime import datetime

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
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print_success(f"{description} completed successfully!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print_error(f"{description} failed!")
            if result.stderr:
                print("Error details:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"{description} timed out after 5 minutes")
        return False
    except Exception as e:
        print_error(f"Error running {script_name}: {str(e)}")
        return False

def check_aws_credentials():
    """Check if AWS credentials are set"""
    aws_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']
    missing_vars = [var for var in aws_vars if not os.environ.get(var)]
    
    if missing_vars:
        print_error("Missing AWS credentials!")
        print_info("Please set these environment variables:")
        for var in missing_vars:
            print_info(f"  - {var}")
        return False
    return True

def main():
    """Main function to run the complete project workflow"""
    print("🎯 Time Series Forecasting Project - Complete Workflow")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 0: Check AWS Configuration
    print_header("STEP 0: AWS CONFIGURATION CHECK")
    if not check_aws_credentials():
        print_error("AWS credentials not found in environment variables")
        print_info("Please set your AWS credentials and try again")
        return False
    
    if not run_script("0_test_aws_setup.py", "AWS Configuration Test"):
        print_error("AWS configuration test failed!")
        print_info("Please fix AWS configuration and try again")
        return False
    
    # Project workflow steps
    steps = [
        ("1_demo_quick_start.py", "Quick Demo (Financial Data Analysis)"),
        ("2_data_collection.py", "Data Collection from Multiple Markets"),
        ("3_feature_engineering.py", "Feature Engineering & Technical Indicators"),
        ("4_regime_detection.py", "Market Regime Detection"),
        ("5_ml_models.py", "Machine Learning Model Training"),
        ("6_full_pipeline.py", "Complete Pipeline Execution")
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