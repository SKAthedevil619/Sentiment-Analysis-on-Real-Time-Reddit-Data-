from transformers import pipeline
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn.metrics import accuracy_score

# Load the tokenizer and model
model = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model)

def expand_contractions(text):
    contractions_pattern = re.compile(r"n\'t", re.IGNORECASE)
    expanded_text = contractions_pattern.sub(" not", text)
    return expanded_text

def preprocess_text(text):
    text = expand_contractions(text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

df = pd.read_csv('modified_dataset.csv', header=None)
df.columns = ['sentiment','Text']

df['PreprocessedText'] = df['Text'].apply(preprocess_text)

# Apply the sentiment analysis model
df['ModelSentiment'] = df['PreprocessedText'].apply(lambda x: sentiment_pipeline(x)[0]['label'])

sentiment_mapping = {
    'LABEL_0': '0',  
    'LABEL_1': '2',
    'LABEL_2': '4'
}
df['ModelSentiment'] = df['ModelSentiment'].map(sentiment_mapping)

print(df[['sentiment','Text', 'ModelSentiment']].head())

print("Saving the DataFrame to CSV...")
df.to_csv('Model_sentiments.csv', index=False)
print("DataFrame saved as 'dataset_with_model_sentiments.csv'.")

# Accuracy percentage
df = pd.read_csv('Model_sentiments.csv')

accuracy = accuracy_score(df.iloc[:, 0], df.iloc[:, -1])
accuracy_percentage = accuracy * 100
print("Accuracy:", f"{accuracy_percentage:.2f}%")