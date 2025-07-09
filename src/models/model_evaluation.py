import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.exceptions import NotFittedError

# ---------------- Logging Configuration ---------------- #
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # ---------------- Load Trained Model ---------------- #
    model_path = 'models/model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    with open(model_path, 'rb') as model_file:
        lr = pickle.load(model_file)
    logging.info("Model loaded successfully.")

except FileNotFoundError as fnf:
    logging.error(fnf)
    raise
except pickle.UnpicklingError as e:
    logging.error(f"Pickle loading error: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error loading the model: {e}")
    raise

try:
    # ---------------- Load Test Data ---------------- #
    test_data_path = './data/interim/test_bow.csv'
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at: {test_data_path}")
    
    test_data = pd.read_csv(test_data_path)
    x_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    logging.info(f"Test data loaded. Shape: X={x_test.shape}, y={y_test.shape}")

except FileNotFoundError as fnf:
    logging.error(fnf)
    raise
except pd.errors.EmptyDataError:
    logging.error("Test CSV file is empty.")
    raise
except pd.errors.ParserError:
    logging.error("Test CSV file is malformed.")
    raise
except Exception as e:
    logging.error(f"Unexpected error loading test data: {e}")
    raise

try:
    # ---------------- Make Predictions ---------------- #
    y_pred = lr.predict(x_test)
    y_pred_prob = lr.predict_proba(x_test)[:, 1]
    logging.info("Predictions generated successfully.")

except NotFittedError:
    logging.error("Model is not fitted. Train it before evaluation.")
    raise


#calculate evaluation metrics 
accuracy = accuracy_score(y_test,y_pred)
precission = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
auc = roc_auc_score(y_test,y_pred_prob)


metrics_dict = {
    'accuracy':accuracy,
    'precission':precission,
    'recall':recall,
    'auc':auc
}

with open('metric.json','w') as file:
    json.dump(metrics_dict,file,indent=4)
