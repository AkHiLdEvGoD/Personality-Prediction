from config.logging_config import get_logger
import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression

logger = get_logger(__name__)

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

def train_model(X, y):
    try:
        model = LogisticRegression(C=0.01,penalty = 'l1',solver = 'liblinear')
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
        X, y = load_data('./local_S3/data/processed/train.csv')
        model = train_model(X, y)
        save_model(model, './local_S3/models')
        logger.info('Model training done and model saved')
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")

if __name__ == '__main__':
    main()