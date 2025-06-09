
print("tushar")
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# Constants
# Constants
DATA_URL = "https://raw.githubusercontent.com/tusharsarodeAI/Tata-Steel-Predictive-Maintenance-XGBoost/refs/heads/master/tata_steel_predictive_maintenance.csv"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


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

# Logger setup
logger = get_logger('data_ingestion')


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a local CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def create_target(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Create binary target column 'maintenance_required' based on Bush_Wear_mm."""
    try:
        df['maintenance_required'] = df['Bush_Wear_mm'].apply(lambda x: 1 if x >= threshold else 0)
        logger.debug("Target column 'maintenance_required' created.")
        return df
    except KeyError as e:
        logger.error(f"Required column not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during target creation: {e}")
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, save_dir: str) -> None:
    """Save the train and test datasets to CSV."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
        logger.debug(f"Train and test data saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving files: {e}")
        raise


# Main execution
if __name__ == "__main__":
    df = load_data(DATA_URL)
    df = create_target(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    save_data(train_df, test_df, LOCAL_DATA_DIR)

