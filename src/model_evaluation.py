from config.logging_config import get_logger
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
import pickle
import mlflow
import mlflow.sklearn
import dagshub
import os
import json
import pandas as pd
import yaml

logger = get_logger(__name__)

mlflow.set_tracking_uri("https://dagshub.com/AkHiLdEvGoD/Personality-Prediction.mlflow")
dagshub.init(repo_owner='AkHiLdEvGoD', repo_name='Personality-Prediction', mlflow=True)

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

def load_model(model_path:str):
    try:
        with open(model_path,'rb') as f:
            model = pickle.load(f)
        logger.info('Model Loaded for evaluation')
        return model
    
    except FileNotFoundError:
        logger.error(f'Model not found at {model_path}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error occured while loading the model {e}')
        raise

def load_data(data_path:str):
    try:
        df = pd.read_csv(data_path)
        logger.info(f'Data loaded from path {data_path}')
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def evaluate_model(model,df:pd.DataFrame):
    try:
        X = df.drop(columns = ['target'])
        y = df['target']

        y_pred = model.predict(X)

        accuracy = accuracy_score(y,y_pred)
        precision = precision_score(y,y_pred)
        recall = recall_score(y,y_pred)
        f1 = f1_score(y,y_pred)

        metric_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'f1_score' : f1
        }

        logger.info('All metrics of model evaluated')
        return metric_dict
    
    except Exception as e:
        logger.error(f'Unexpected error occured while evaluating model {e}')
        raise

def save_metrics(metric_dict,save_path):
    try:
        os.makedirs(save_path,exist_ok=True)
        metric_save_path = os.path.join(save_path,'metrics.json')
        with open(metric_save_path,'w') as f:
            json.dump(metric_dict,f,indent=4)
        logger.info(f'Metrics saved at {metric_save_path}')
    
    except Exception as e:
        logger.error(f'Unexpected error occured while saving metrics : {e}')
        raise

def save_model_info(run_id,model_path,file_path):
    try:
        os.makedirs(file_path,exist_ok=True)
        info_file_path = os.path.join(file_path,'model_info.json')
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(info_file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f'Model info saved to {file_path}')
    except Exception as e:
        logger.error(f'Error occurred while saving the model info: {e}')
        raise

def main():
    mlflow.set_experiment('dvc_pipeline')
    with mlflow.start_run() as run:
        try:
            params = load_params('./config/params.yaml')
            model = load_model('./local_S3/models/trained_model.pkl')
            df = load_data('./local_S3/data/processed/test.csv')

            metrics = evaluate_model(model,df)

            save_metrics(metrics,'./local_S3/metrics')

            mlflow.log_metrics(metrics)
            
            model_type = params['model_training']['model_type']
            mlflow.log_param('Model_type',model_type)

            if hasattr(model,'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            mlflow.sklearn.log_model(model,f'{model_type}')

            mlflow.log_artifact('./local_S3/metrics/metrics.json')

            save_model_info(run.info.run_id,'./local_S3/models','./local_S3/model_info')
            
            logger.info('Model Evaluation logged and Completed')

        except Exception as e:
            logger.error(f'Unexpected error occure during Model Evaluation : {e}')
    
if __name__ == '__main__':
    main()






    