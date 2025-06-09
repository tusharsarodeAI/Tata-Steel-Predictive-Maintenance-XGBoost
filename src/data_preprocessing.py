import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


# logger_config.py
import logging
import os

def get_logger(name):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:  # Prevent adding handlers multiple times
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = get_logger('data_preprocessing')




def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by combining columns, cleaning text,
    handling missing values, and label encoding categorical variables.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Log null values
        nullvalues = df.isnull().sum()
        logger.debug(f'Null values:\n{nullvalues}')

        df.dropna()

       

        

        logger.debug('Preprocessing complete')
        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during preprocessing: %s', e)
        raise


def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(project_root, 'data', 'raw', 'dataset.csv')
        

        dataset = pd.read_csv(train_path)
        

        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(dataset)

        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "dataset_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()