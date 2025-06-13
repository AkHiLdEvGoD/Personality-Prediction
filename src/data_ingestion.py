from config.logging_config import get_logger
import pandas as pd
import os

logger = get_logger(__name__)

def load_data(data_url:str):
    try:
        df = pd.read_csv(data_url)
        logger.info(f'Data loaded from {data_url}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file : {e}')
        raise
    except Exception as e:
        logger.error(f'An unexpected error occured while loading data {e}')
        raise

def preprocessing(df:pd.DataFrame):
    try:
        logger.info('Pre-processing ...')
        final_df = df.drop_duplicates()

        num_cols = final_df.select_dtypes(include='number').columns
        cat_cols = final_df.select_dtypes(include='object').columns
        for i in num_cols:
            final_df.loc[:,i] = final_df[i].fillna(df[i].median())

        for i in cat_cols:
            final_df.loc[:,i] = final_df[i].fillna(df[i].mode()[0])

        logger.info(f'Data Shape after preprocessing : {final_df.shape}')
        logger.info('Data Preprocessing Completed')
        return final_df
    
    except KeyError as e:
        logger.error(f'Missing column in dataframe : {e}')
        raise
    
    except Exception as e:
        logger.error(f'An unexpected error occured during preprocessing : {e}') 
        raise

def save_data(df:pd.DataFrame,destination_path:str):
    try:
        raw_data_path = os.path.join(destination_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        logger.info(f'Saving Data to {raw_data_path}')
        df.to_csv(os.path.join(raw_data_path,'raw_data.csv'),index=False)
        logger.debug(f'Raw Processed data saved to {raw_data_path}')

    except Exception as e:
        logger.error(f'An unexpected error occured while saving the data : {e}')
        raise

def main():
    try:
        df = load_data('C:/Users/akhde/OneDrive/Desktop/MLOps/Loan-Fraud-Detection/data/raw/personality_dataset.csv')
        final_df = preprocessing(df)
        save_data(final_df,destination_path='./local_S3/data')
        logger.info('Data Ingestion Completed')

    except Exception as e:
        logger.error(f'Failed to complete data ingestion process : {e}')
        print('error',e)

if __name__ == '__main__':
    main()