import numpy as np
import pandas as pd
import logging
import os 
import re 
import nltk
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('data_preprocessing.log')


formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)


logger.addHandler(console_handler)
logger.addHandler(file_handler)



#fetch the data from data/raw
try:
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/train.csv')
except Exception as e:
    raise Exception(f"Error while reading data from raw file")

#transform the data
def clean_data(text):
    try:
        text = text.lower()
        text = text.translate(str.maketrans('','',string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        return text
    except Exception as e:
        raise Exception(f"Error while cleaning data: {e}")

def noise_removal(text):
    try:  
        text = re.sub(r"http\S+|www\S+|https\S+",'',text)
        text = re.sub(r'[^A-Za-z\s]','',text)
        return text
    except Exception as e:
        raise Exception(f"Error while removing noise from data: {e}")

def tokenization(text):
    try:
        tokens = word_tokenize(text)
        return tokens
    except Exception as e:
        raise Exception(f"Error while tokenizing the text:{e}")

def stopword_removal(tokens):
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return tokens
    except Exception as e:
        raise Exception(f"Error while removing stopword from text:{e}")

def lemmatization(tokens):
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens
    except Exception as e:
        raise Exception(f"Error while lemmatizing the text: {e}")


def preprocess_text(df):
    try:
        
        df['content'] = df['content'].apply(lambda content: clean_data(content))
        df['content'] = df['content'].apply(lambda content: noise_removal(content))
        df['content'] = df['content'].apply(lambda content:tokenization(content))
        df['content'] = df['content'].apply(lambda content: stopword_removal(content))
        df['content'] = df['content'].apply(lambda content: lemmatization(content))
        df['content'] = df['content'].apply(lambda tokens: ' '.join(tokens))
        return df 
    except Exception as e:
        raise Exception(f"Error while preprocess text : {e}")
        

train_processed_data = preprocess_text(train_data)
test_processed_data = preprocess_text(test_data)

#store the data inside data/processed

data_path = os.path.join("data","processed")

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"))


