import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import logging

# Logger setup
def get_logger(name):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

logger = get_logger('model_training')


def load_data(filepath):
    logger.debug(f'Loading DataFrame from: {filepath}')
    df = pd.read_csv(filepath)

    X = df.drop("maintenance_required", axis=1)
    Y = df["maintenance_required"]

    logger.debug(f'Features shape: {X.shape}, Labels shape: {Y.shape}')
    return X, Y


def train_model(X, Y):
    """
    Train an XGBoost model on the preprocessed data.
    """
    try:
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, Y, test_size=0.30, random_state=42
        )
        logger.debug(f'Train shape: {X_train.shape}, Validation shape: {X_val.shape}')

        # Train model
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        logger.debug('XGBoost model training complete.')

        # Evaluate
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logger.debug(f'Validation Accuracy: {acc:.4f}')
        logger.debug('Classification Report:\n' + classification_report(y_val, y_pred))

        return model

    except Exception as e:
        logger.error(f'Error during model training: {e}')
        raise


def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    logger.debug(f'Model saved to: {output_path}')


def main():
    try:
        features_path = os.path.abspath(os.path.join('data', 'processed', 'dataset_processed.csv'))
        model_path = os.path.join('models', 'xgb_model.pkl')

        logger.debug(f'Feature path: {features_path}')
        logger.debug(f'Model path: {model_path}')

        X, Y = load_data(features_path)
        model = train_model(X, Y)
        save_model(model, model_path)

    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
    except Exception as e:
        logger.error(f'Model training failed: {e}')


if __name__ == '__main__':
    main()
