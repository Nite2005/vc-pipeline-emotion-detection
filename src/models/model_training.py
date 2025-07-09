import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

# ------------------ Logging Configuration ------------------ #
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # ------------------ Load Training Data ------------------ #
    train_data_path = './data/interim/train_bow.csv'
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at: {train_data_path}")
    
    train_data = pd.read_csv(train_data_path)
    logging.info("Training data loaded successfully.")

    # ------------------ Extract Features and Labels ------------------ #
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    logging.info(f"Extracted features and labels from training data. Shape: X={x_train.shape}, y={y_train.shape}")

except FileNotFoundError as fnf:
    logging.error(fnf)
    raise
except pd.errors.EmptyDataError:
    logging.error("Training CSV file is empty.")
    raise
except pd.errors.ParserError:
    logging.error("Training CSV file is corrupted or not properly formatted.")
    raise
except Exception as e:
    logging.error(f"Unexpected error while loading training data: {e}")
    raise

try:
    # ------------------ Train Logistic Regression Model ------------------ #
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    logging.info("Logistic Regression model trained successfully.")

except ValueError as ve:
    logging.error(f"ValueError during model training: {ve}")
    raise
except Exception as e:
    logging.error(f"Unexpected error during model training: {e}")
    raise

try:
    # ------------------ Save the Model ------------------ #
    with open('models/model.pkl', 'wb') as model_file:
        pickle.dump(lr, model_file)
    logging.info("Model saved successfully to model.pkl")

except (IOError, pickle.PickleError) as e:
    logging.error(f"Error while saving the model: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error during model saving: {e}")
    raise
