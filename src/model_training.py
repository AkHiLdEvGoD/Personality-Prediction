from config.logging_config import get_logger
import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import yaml

logger = get_logger(__name__)

def load_params(params_path:str):
    try:
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
        logger.info(f'Parameter retrieved from {params_path}')
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(train_path):
    try:
        df = pd.read_csv(train_path)
        X = df.drop(columns=['target'])
        y = df['target']
        logger.info(f"Train data loaded with shape: {df.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def train_model(X,y,params):
    try:
        model_type = params['model_training']['model_type']

        if model_type == 'logistic_regression':
            model_params = params['model_training']['logistic_regression']
            model = LogisticRegression(**model_params)
        
        elif model_type == "svc":
            model_params = params['model_training']["svc"]
            model = SVC(**model_params)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        model.fit(X, y)
        logger.info("Model training completed")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model,save_path:str):
    try:
        os.makedirs(save_path,exist_ok=True)
        model_path = os.path.join(save_path,'trained_model.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info('Model saved to %s', model_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params('./params.yaml')
        X, y = load_data('./local_Storage/data/processed/train.csv')
        model = train_model(X,y,params)
        save_model(model, './local_Storage/models')
        logger.info('Model training done and model saved')
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")

if __name__ == '__main__':
    main()