# AWS SageMaker Deployment for Time Series Forecasting
import boto3
import pandas as pd
import numpy as np
import json
import logging
import pickle
from datetime import datetime
import os
import tarfile

from config import configure_aws_environment, AWS_REGION, SAGEMAKER_ROLE, AWS_ACCOUNT_ID
from ml_models import train_all_models

# Configure AWS
configure_aws_environment()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SageMakerModelDeployer:
    """Deploy and manage ML models on AWS SageMaker"""
    
    def __init__(self):
        self.sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
        self.runtime_client = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
        
        # Create S3 bucket for models
        self.bucket_name = f'timeseries-models-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        self.create_s3_bucket()
        
        self.endpoints = {}
        
    def create_s3_bucket(self):
        """Create S3 bucket for model artifacts"""
        try:
            if AWS_REGION == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                )
            logger.info(f"Created S3 bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error creating S3 bucket: {e}")
            # Try to use existing bucket
            self.bucket_name = 'default-timeseries-models'
    
    def package_model(self, model, model_name, model_type='sklearn'):
        """Package model for SageMaker deployment"""
        logger.info(f"Packaging {model_name} model...")
        
        # Create model directory
        model_dir = f"models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create inference script
        inference_script = self.create_inference_script(model_name, model_type)
        with open(f"{model_dir}/inference.py", 'w') as f:
            f.write(inference_script)
        
        # Create requirements file
        requirements = self.create_requirements(model_type)
        with open(f"{model_dir}/requirements.txt", 'w') as f:
            f.write(requirements)
        
        # Create tarball
        tarball_path = f"{model_dir}.tar.gz"
        with tarfile.open(tarball_path, 'w:gz') as tar:
            tar.add(model_dir, arcname='.')
        
        # Upload to S3
        s3_key = f"models/{model_name}.tar.gz"
        self.s3_client.upload_file(tarball_path, self.bucket_name, s3_key)
        
        model_url = f"s3://{self.bucket_name}/{s3_key}"
        logger.info(f"Model uploaded to {model_url}")
        
        return model_url
    
    def create_inference_script(self, model_name, model_type):
        """Create inference script for SageMaker"""
        script = f'''
import pickle
import pandas as pd
import numpy as np
import json
import logging
from io import StringIO

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load model from directory"""
    with open(f"{{model_dir}}/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    elif request_content_type == "text/csv":
        return pd.read_csv(StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {{request_content_type}}")

def predict_fn(input_data, model):
    """Make predictions"""
    try:
        if hasattr(model, 'predict'):
            if '{model_name}' == 'arima':
                # ARIMA prediction
                predictions = model.predict(steps=len(input_data))
                if len(predictions) != len(input_data):
                    predictions = np.full(len(input_data), input_data.mean() if len(input_data) > 0 else 0)
            else:
                # XGBoost or LSTM prediction
                predictions = model.predict(input_data)
        else:
            predictions = np.zeros(len(input_data))
        
        return predictions.tolist()
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        return [0.0] * len(input_data)

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == "application/json":
        return json.dumps({{"predictions": prediction}})
    else:
        return str(prediction)
'''
        return script
    
    def create_requirements(self, model_type):
        """Create requirements.txt for model"""
        base_requirements = [
            "pandas==2.0.3",
            "numpy==1.24.3",
            "scikit-learn==1.3.0",
            "joblib==1.3.2"
        ]
        
        if model_type == 'xgboost':
            base_requirements.append("xgboost==1.7.6")
        elif model_type == 'arima':
            base_requirements.append("statsmodels==0.14.0")
        elif model_type == 'lstm':
            base_requirements.extend(["tensorflow==2.13.0", "keras==2.13.1"])
        
        return '\n'.join(base_requirements)
    
    def create_model(self, model_name, model_url, role_arn):
        """Create SageMaker model"""
        model_name_sagemaker = f"{model_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Use appropriate container image based on model type
        if 'xgboost' in model_name:
            image_uri = f"246618743249.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-xgboost:1.5-1"
        else:
            image_uri = f"246618743249.dkr.ecr.{AWS_REGION}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
        
        try:
            response = self.sagemaker_client.create_model(
                ModelName=model_name_sagemaker,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_url,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': model_url
                    }
                },
                ExecutionRoleArn=role_arn
            )
            logger.info(f"Created SageMaker model: {model_name_sagemaker}")
            return model_name_sagemaker
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None
    
    def create_endpoint_config(self, model_name, config_name):
        """Create endpoint configuration"""
        try:
            response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.t2.medium',
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            logger.info(f"Created endpoint config: {config_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating endpoint config: {e}")
            return False
    
    def create_endpoint(self, endpoint_name, config_name):
        """Create SageMaker endpoint"""
        try:
            response = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            logger.info(f"Creating endpoint: {endpoint_name}")
            
            # Wait for endpoint to be in service
            waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name, WaiterConfig={'Delay': 30, 'MaxAttempts': 20})
            
            logger.info(f"Endpoint {endpoint_name} is now in service")
            return True
        except Exception as e:
            logger.error(f"Error creating endpoint: {e}")
            return False
    
    def deploy_models(self, models):
        """Deploy all trained models to SageMaker"""
        logger.info("Starting model deployment to SageMaker...")
        
        role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{SAGEMAKER_ROLE}"
        
        for model_name, model in models.items():
            try:
                logger.info(f"Deploying {model_name} model...")
                
                # Determine model type
                if model_name == 'xgboost':
                    model_type = 'xgboost'
                elif model_name == 'arima':
                    model_type = 'arima'
                elif model_name == 'lstm':
                    model_type = 'lstm'
                else:
                    model_type = 'sklearn'
                
                # Package model
                model_url = self.package_model(model, model_name, model_type)
                
                # Create SageMaker model
                sm_model_name = self.create_model(model_name, model_url, role_arn)
                if not sm_model_name:
                    continue
                
                # Create endpoint configuration
                config_name = f"{model_name}-config-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                if not self.create_endpoint_config(sm_model_name, config_name):
                    continue
                
                # Create endpoint
                endpoint_name = f"{model_name}-endpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                if self.create_endpoint(endpoint_name, config_name):
                    self.endpoints[model_name] = endpoint_name
                    logger.info(f"✅ {model_name} model deployed successfully")
                else:
                    logger.error(f"❌ Failed to deploy {model_name} model")
                    
            except Exception as e:
                logger.error(f"Error deploying {model_name}: {e}")
        
        logger.info(f"Deployment complete. Active endpoints: {list(self.endpoints.keys())}")
        return self.endpoints
    
    def predict(self, model_name, data, content_type="application/json"):
        """Make predictions using deployed model"""
        if model_name not in self.endpoints:
            logger.error(f"Model {model_name} not deployed")
            return None
        
        endpoint_name = self.endpoints[model_name]
        
        try:
            if content_type == "application/json":
                payload = json.dumps(data.to_dict('records'))
            else:
                payload = data.to_csv(index=False)
            
            response = self.runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=payload
            )
            
            result = json.loads(response['Body'].read().decode())
            return result['predictions']
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

class RegimeAwarePredictionRouter:
    """Route predictions to appropriate models based on detected regime"""
    
    def __init__(self, deployer):
        self.deployer = deployer
        self.regime_model_mapping = {
            'High_Vol': 'lstm',      # Use LSTM for high volatility
            'Medium_Vol': 'xgboost', # Use XGBoost for medium volatility
            'Low_Vol': 'arima',      # Use ARIMA for low volatility
            'Uptrend': 'xgboost',    # Use XGBoost for trending markets
            'Downtrend': 'lstm',     # Use LSTM for downtrends
            'Sideways': 'arima'      # Use ARIMA for sideways markets
        }
    
    def detect_current_regime(self, data):
        """Detect current market regime from recent data"""
        # Simple regime detection based on recent volatility and trend
        if 'volatility_20' in data.columns:
            recent_vol = data['volatility_20'].tail(5).mean()
            if recent_vol > 0.03:
                return 'High_Vol'
            elif recent_vol > 0.01:
                return 'Medium_Vol'
            else:
                return 'Low_Vol'
        
        # Fallback to returns-based detection
        if 'returns' in data.columns:
            recent_returns = data['returns'].tail(10)
            if recent_returns.std() > 0.02:
                return 'High_Vol'
            elif recent_returns.mean() > 0.001:
                return 'Uptrend'
            elif recent_returns.mean() < -0.001:
                return 'Downtrend'
            else:
                return 'Sideways'
        
        return 'Medium_Vol'  # Default
    
    def route_prediction(self, data):
        """Route prediction to appropriate model based on regime"""
        regime = self.detect_current_regime(data)
        model_name = self.regime_model_mapping.get(regime, 'xgboost')
        
        logger.info(f"Detected regime: {regime}, routing to {model_name} model")
        
        prediction = self.deployer.predict(model_name, data)
        
        return {
            'prediction': prediction,
            'regime': regime,
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main function to deploy models"""
    logger.info("Starting AWS SageMaker deployment...")
    
    try:
        # Load trained models
        data_files = [
            f"data/data_with_regimes.csv",
            f"data/enhanced_features.csv",
            f"data/raw_financial_data.csv"
        ]
        
        data = None
        for file_path in data_files:
            try:
                data = pd.read_csv(file_path)
                data['date'] = pd.to_datetime(data['date'])
                logger.info(f"Loaded data from {file_path}")
                break
            except FileNotFoundError:
                continue
        
        if data is None:
            logger.error("No data found. Please run the pipeline first.")
            return
        
        # Train models if not already available
        logger.info("Training models for deployment...")
        models, results = train_all_models(data)
        
        if not models:
            logger.error("No models trained successfully")
            return
        
        # Deploy to SageMaker
        deployer = SageMakerModelDeployer()
        endpoints = deployer.deploy_models(models)
        
        if endpoints:
            logger.info("✅ Models deployed successfully!")
            
            # Test regime-aware routing
            router = RegimeAwarePredictionRouter(deployer)
            
            # Test prediction with sample data
            sample_data = data.tail(10).select_dtypes(include=[np.number]).fillna(0)
            test_result = router.route_prediction(sample_data)
            
            print("\n" + "="*60)
            print("SAGEMAKER DEPLOYMENT SUMMARY")
            print("="*60)
            print(f"Deployed endpoints: {list(endpoints.keys())}")
            print(f"S3 bucket: {deployer.bucket_name}")
            print(f"Test prediction result: {test_result}")
            print("="*60)
            
            # Save deployment info
            deployment_info = {
                'endpoints': endpoints,
                'bucket': deployer.bucket_name,
                'deployment_time': datetime.now().isoformat(),
                'test_result': test_result
            }
            
            with open('results/sagemaker_deployment.json', 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            logger.info("Deployment info saved to results/sagemaker_deployment.json")
        else:
            logger.error("❌ Model deployment failed")
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

if __name__ == "__main__":
    main() 