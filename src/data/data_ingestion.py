import numpy as np
import pandas as pd
import yaml
import os
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')


formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)


logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            config = yaml.safe_load(file)
        test_size = config['Data_ingestion']['test_size']
        logger.debug('test_size reterived')
        return test_size
    except FileNotFoundError:
        logger.error('filenotfound error')
        raise 
    except KeyError as e:
        logger.error('yaml error')
        raise 
    except Exception as e:
        logger.error('some error occured')
        raise 


def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        raise Exception(f"Error while reading data from {url}: {e}")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'tweet_id' in df.columns:
            df.drop(columns=['tweet_id'], inplace=True)
        
        if 'sentiment' not in df.columns:
            raise ValueError("Column 'sentiment' not found in data")

        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()

        if final_df.empty:
            raise ValueError("No data with 'happiness' or 'sadness' sentiment found")

        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

        return final_df
    except Exception as e:
        raise Exception(f"Error while processing data: {e}")


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    try:
        data_path = os.path.join("data", "raw")
        os.makedirs(data_path, exist_ok=True)

        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        raise Exception(f"Error while saving data: {e}")


def main():
    try:
        test_size = load_params('params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data)
        print("Data ingestion and processing completed successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
