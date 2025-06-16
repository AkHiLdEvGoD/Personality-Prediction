from config.logging_config import get_logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib
import os
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


def make_derieved_features(df:pd.DataFrame):
    try:
        df['Offline_social_activity'] = df['Social_event_attendance'] * df['Going_outside']
        logger.info(f'Derived feature added. Shape of dataframe : {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Unexpected error occured while making derived features : {e}')
        raise


def preprocess_and_split(df:pd.DataFrame,target_col:str,test_size:float,save_dir:str):
    try:
        os.makedirs(save_dir,exist_ok=True)
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)

        num_cols = X.select_dtypes(include='number').columns.to_list()
        cat_cols = X.select_dtypes(include=['object']).columns.to_list()

        preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num',StandardScaler(),num_cols),
                ('cat',OrdinalEncoder(),cat_cols)
            ]
        )

        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        joblib.dump(preprocessing_pipeline,os.path.join(save_dir,'preprocessing_pipeline.pkl'))
        joblib.dump(le,os.path.join(save_dir,'label_encoder.pkl'))
        logger.info(f'Preprocessing_pipeline saved to path : {save_dir}')

        logger.info('Preprocessing and splitting done.')
        return X_train_processed,X_test_processed,y_train,y_test
    
    except KeyError as e:
        logger.error(f'Missing column in dataframe : {e}')
        raise
    
    except Exception as e:
        logger.error(f'Unexpected error occured while Encoding Categorical features : {e}')
        raise

def save_preprocessed_data(X_train,X_test,y_train,y_test,destination_path:str):
    try:
        # train_df = pd.concat([pd.DataFrame(X_train),pd.Series(y_train,name='target')],axis=1)
        # test_df = pd.concat([pd.DataFrame(X_test),pd.Series(y_test,name='target')],axis=1)

        X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
        y_train_series = pd.Series(y_train, name='target').reset_index(drop=True)

        train_df = pd.concat([X_train_df, y_train_series], axis=1)

        X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
        y_test_series = pd.Series(y_test, name='target').reset_index(drop=True)

        test_df = pd.concat([X_test_df, y_test_series], axis=1)

        processed_data_path = os.path.join(destination_path,'processed')
        os.makedirs(processed_data_path,exist_ok=True)
        train_df.to_csv(os.path.join(processed_data_path,'train.csv'),index=False)
        test_df.to_csv(os.path.join(processed_data_path,'test.csv'),index=False)
        logger.info(f'Train and test data saved to path : {processed_data_path}')
        logger.debug(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    except Exception as e:
        logger.error(f'Unexpected error occured while saving preprocessed data : {e}')
        raise

def main():
    try:
        params = load_params(params_path='./params.yaml')
        test_size = params['data_preprocessing']['test_size']

        df = pd.read_csv('./local_Storage/data/raw/raw_data.csv')


        featured_df = make_derieved_features(df)
        X_train,X_test,y_train,y_test = preprocess_and_split(featured_df,'Personality',test_size,'./local_Storage/models')
        save_preprocessed_data(X_train,X_test,y_train,y_test,destination_path='./local_Storage/data/')
        logger.info('Data Preprocessing Completed')
    
    except Exception as e:
        logger.error(f'Failed to complete data ingestion process : {e}')

if __name__ == '__main__':
    main()


