import os
import yaml
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- Logging Configuration ---------------- #
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # ---------------- Load max_features from params.yaml ---------------- #
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
        max_feature = params['feature_engineering']['max_features']
    logging.info(f"Loaded max_features: {max_feature}")
except FileNotFoundError:
    logging.error("params.yaml not found.")
    raise
except yaml.YAMLError as e:
    logging.error(f"Error parsing YAML: {e}")
    raise
except KeyError:
    logging.error("Key 'feature_engineering -> max_features' not found in params.yaml")
    raise
except Exception as e:
    logging.error(f"Unexpected error while reading params.yaml: {e}")
    raise

try:
    # ---------------- Load processed data ---------------- #
    train_data = pd.read_csv('./data/processed/train_processed.csv')
    test_data = pd.read_csv('./data/processed/test_processed.csv')
    logging.info("Train and test data loaded successfully.")
    
    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)
except FileNotFoundError as fnf:
    logging.error(f"Data file not found: {fnf}")
    raise
except pd.errors.EmptyDataError:
    logging.error("One of the CSV files is empty.")
    raise
except pd.errors.ParserError:
    logging.error("Error parsing CSV files.")
    raise
except Exception as e:
    logging.error(f"Unexpected error loading CSVs: {e}")
    raise

try:
    # ---------------- Extract features and labels ---------------- #
    x_train = train_data['content'].values
    x_test = test_data['content'].values

    y_train = train_data['sentiment'].values
    y_test = test_data['sentiment'].values
    logging.info("Extracted features and labels from data.")
except KeyError as e:
    logging.error(f"Missing expected column: {e}")
    raise

try:
    # ---------------- Apply Bag of Words ---------------- #
    vectorizer = CountVectorizer(max_features=max_feature)
    
    X_train_bow = vectorizer.fit_transform(x_train)
    X_test_bow = vectorizer.transform(x_test)  # Use transform, not fit_transform for test
    logging.info("BOW transformation successful.")
except Exception as e:
    logging.error(f"Error in CountVectorizer: {e}")
    raise

try:
    # ---------------- Create DataFrames ---------------- #
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test
    logging.info("Train and test DataFrames created.")
except Exception as e:
    logging.error(f"Error creating DataFrames: {e}")
    raise

try:
    # ---------------- Save to CSV ---------------- #
    data_path = os.path.join('data', 'interim')
    os.makedirs(data_path, exist_ok=True)

    train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
    logging.info("BOW feature files saved successfully.")
except Exception as e:
    logging.error(f"Error saving CSV files: {e}")
    raise
