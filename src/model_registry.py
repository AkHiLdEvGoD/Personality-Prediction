from config.logging_config import get_logger
import json
import mlflow
import dagshub
import os
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

logger = get_logger(__name__)

# mlflow.set_tracking_uri("https://dagshub.com/AkHiLdEvGoD/Personality-Prediction.mlflow")
# dagshub.init(repo_owner='AkHiLdEvGoD', repo_name='Personality-Prediction', mlflow=True)

dagshub_token = os.getenv("PERSONALITY_TEST")
if not dagshub_token:
    raise EnvironmentError("PERSONALITY_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "AkHiLdEvGoD"
repo_name = "Personality-Prediction"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def load_model_info(file_path: str):
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug(f'Model info loaded from {file_path}')
        return model_info
    except FileNotFoundError:
        logger.error(f'File not found: {file_path}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the model info: {e}')
        raise


def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/model"
        
        model_version = mlflow.register_model(model_uri, model_name)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    
    except Exception as e:
        logger.error(f'Error during model registration: {e}')
        raise

def main():
    try:
        model_info_path = './local_S3/model_info/model_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error(f'Failed to complete the model registration process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()