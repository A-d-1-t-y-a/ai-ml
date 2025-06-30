# AWS Setup Guide for Time Series Forecasting Project

## Quick Setup Instructions

### 1. Install AWS CLI (if not already installed)
Download from: https://aws.amazon.com/cli/
Or using PowerShell:
```powershell
winget install Amazon.AWSCLI
```

### 2. Configure AWS Credentials
Run this command and follow the prompts:
```bash
aws configure
```

You'll need:
- AWS Access Key ID
- AWS Secret Access Key  
- Default region (e.g., us-east-1)
- Default output format (json)

### 3. Alternative: Environment Variables
Set these in your system or .env file:
```
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
```

### 4. Test Your Configuration
Run the test script:
```bash
python test_aws_config.py
```

### 5. For S3 Storage (Optional)
If you want to use S3 for data storage:
- Create an S3 bucket in AWS Console
- Update the S3_BUCKET_NAME in config.py

### 6. For SageMaker (Optional)
If you want to use SageMaker:
- Ensure you have SageMaker permissions
- Create a SageMaker execution role

## Notes:
- The project works without AWS (data stored locally)
- AWS is only needed for cloud storage and SageMaker features
- You can run the quick_start.py demo without AWS
