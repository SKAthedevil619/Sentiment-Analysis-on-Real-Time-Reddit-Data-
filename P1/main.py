from flask import Flask, render_template, request
import re
import praw
import pandas as pd
from langdetect import detect
from googletrans import Translator
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import prawcore

# Initialize the Translator

app = Flask(__name__)

# Dummy data for fields
fields = ['Entertainment', 'Technology', 'Politics']

# Specify the sentiment analysis model
model = "cardiffnlp/twitter-roberta-base-sentiment"

# Initialize the sentiment analysis pipeline
model_name_english = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline_english = pipeline("sentiment-analysis", model=model_name_english)

# Initialize a separate sentiment pipeline for Hindi texts
model_name_hindi = "bert-base-multilingual-cased"  # Example model that supports Hindi
sentiment_pipeline_hindi = pipeline("sentiment-analysis", model=model_name_hindi)

translator = Translator()

reddit = praw.Reddit(client_id="7nfY244NGfMh7NiWNUK2-Q", client_secret="hKdi5-tImtt-I9WCuFQHchgKZY7F4A",
                     user_agent="Somya Agarwal")

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None


def translate_to_english(text):
    try:
        return translator.translate(text, src='hi', dest='en').text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text


def preprocess_text(text):
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove user mentions
    return text


def classify_language(text):
    try:
        language = detect(text)
        return language
    except:
        return 'unknown'


def translate_comments(comments_df, translator):
    translated_comments = []
    for index, row in comments_df.iterrows():
        if row['language'] != 'en':
            translated_text = translator.translate(row['text'], src=row['language'], dest='en').text
        else:
            translated_text = row['text']
        translated_comments.append({'language': row['language'], 'text': translated_text})
    return pd.DataFrame(translated_comments)


def preprocess_text_1(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def preprocess_comments(comments_df):
    preprocessed_comments = []
    for index, row in comments_df.iterrows():
        preprocessed_text = preprocess_text_1(row['text'])
        preprocessed_comments.append({'language': row['language'], 'text': preprocessed_text})
    return pd.DataFrame(preprocessed_comments)


def separate_hindi_english(comments_df):
    hindi_comments_df = comments_df[comments_df['language'] == 'hi']
    english_comments_df = comments_df[comments_df['language'] == 'en']
    return hindi_comments_df, english_comments_df


def separate_roman_hindi(english_comments_df, hindi_words_set):
    roman_hindi_df = pd.DataFrame(columns=english_comments_df.columns)
    pure_english_df = pd.DataFrame(columns=english_comments_df.columns)
    for index, row in english_comments_df.iterrows():
        text_to_check = row['text'].lower()
        if any(word in text_to_check.split() for word in hindi_words_set):
            roman_hindi_df = pd.concat([roman_hindi_df, pd.DataFrame([row])], ignore_index=True)
        else:
            pure_english_df = pd.concat([pure_english_df, pd.DataFrame([row])], ignore_index=True)
    return roman_hindi_df, pure_english_df


def collect_comments(topic, subreddits, limit_per_subreddit=5):
    comments_data = []
    retry_delay = 60  # Start with a minute delay

    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(topic, sort='top', limit=limit_per_subreddit):
                try:
                    # Limit the replace_more to reduce number of requests
                    submission.comments.replace_more(limit=1)
                    for comment in submission.comments.list()[:10]:  # Limit number of comments processed
                        comment_text = comment.body.strip()
                        comment_language = classify_language(comment_text)
                        if comment_language in ['en', 'hi']:
                            comments_data.append({'language': comment_language, 'text': comment_text})
                except prawcore.exceptions.RequestException:
                    print("Request failed, sleeping for retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        except prawcore.exceptions.TooManyRequests as e:
            print(f"Rate limit exceeded, sleeping for {retry_delay} seconds.")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    return pd.DataFrame(comments_data)

def convert_label(original_label):
    label_mapping = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive'
    }
    return label_mapping.get(original_label, 'Unknown')

def analyze_sentiment_english(df):
    sentiments = []
    for text in df['text']:
        try:
            truncated_text = text[:512]
            result = sentiment_pipeline_english(truncated_text)[0]
            sentiments.append(convert_label(result['label']))
        except Exception as e:
            print(f"Error processing text: {e}")
            sentiments.append('Error')
    df['sentiment'] = sentiments
    return df


def analyze_sentiment_hindi(df):
    sentiments = []
    for text in df['text']:
        try:
            truncated_text = text[:512]
            result = sentiment_pipeline_hindi(truncated_text)[0]
            sentiments.append(convert_label(result['label']))
        except Exception as e:
            print(f"Error processing text: {e}")
            sentiments.append('Error')
    df['sentiment'] = sentiments
    return df


static_directory = r'C:\Users\hp\Documents\Minor Project\P1\static'

def aggregate_sentiments(dataframes_to_combine):
    if not dataframes_to_combine:
        print("No data available for analysis.")
        return pd.DataFrame(), {}  # Ensure an empty DataFrame and dict are returned if no data

    combined_df = pd.concat(dataframes_to_combine)
    if combined_df.empty:
        print("Combined DataFrame is empty.")
        return pd.DataFrame(), {}

    sentiment_counts = combined_df.groupby(['source', 'sentiment']).size().reset_index(name='counts')
    total_counts = combined_df['sentiment'].count()
    sentiment_counts['percentage'] = (sentiment_counts['counts'] / total_counts) * 100

    total_percentages = sentiment_counts.groupby('sentiment')['percentage'].sum().to_dict()  # Use sum to ensure total is 100%
    return sentiment_counts, total_percentages


def plot_combined_sentiment_distribution(sentiment_counts, title, filename):
    print("Debug: DataFrame heading to plot", sentiment_counts.head())  # Debug print to see the DataFrame structure
    if sentiment_counts.empty:
        print("Skipping sentiment analysis due to lack of data.")
        return

    try:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='sentiment', y='percentage', hue='source', data=sentiment_counts, palette='viridis')
        plt.title(title)
        plt.xlabel('Sentiment')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        plt.legend(title='Dataset')
        plt.tight_layout()

        filepath = os.path.join(static_directory, filename)
        plt.savefig(filepath)
        plt.close()
    except Exception as e:
        print(f"An error occurred while plotting: {e}")  # Print any other errors that might occur


def save_plot(dataframe, title, filename):
    plt.figure(figsize=(10, 6))
    sentiment_counts = dataframe['sentiment'].value_counts(normalize=True) * 100
    sentiment_counts = sentiment_counts.reset_index()
    sentiment_counts.columns = ['sentiment', 'percentage']

    sns.barplot(x='sentiment', y='percentage', data=sentiment_counts, palette='viridis')
    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()

    filepath = os.path.join(static_directory, filename)
    plt.savefig(filepath)
    plt.close()

hindi_words_df = pd.read_csv("romanhindi_data.csv")
hindi_words_set = set(hindi_words_df['Word'].str.lower())

def is_roman_hindi(text, hindi_words_set):
    words = set(text.lower().split())
    return any(word in hindi_words_set for word in words)


def analyze_sentiment(text):
    detected_language = detect_language(text)
    print(f"Detected language: {detected_language}")  # Debugging print

    translator = Translator()
    detected_language = detect(text)  # You might want to ensure this line isn't redundant with the previous detection

    # Check if the text is Roman Hindi
    if detected_language == 'en' and is_roman_hindi(text, hindi_words_set):
        detected_language = 'hi'  # Treat Roman Hindi as Hindi for processing

    if detected_language == 'hi':
        # Translate Hindi or Roman Hindi to English
        original_text = text  # Keep the original text for debugging
        text = translator.translate(text, src='hi', dest='en').text
        print(f"Original text: {original_text}")  # Debugging print
        print(f"Translated text: {text}")  # Debugging print

    # After translation, or if the text is originally in English, proceed with the English model
    preprocessed_text = preprocess_text(text)
    print(f"Preprocessed text: {preprocessed_text}")  # Debugging print
    sentiment_result = sentiment_pipeline_english([preprocessed_text])[0]  # Assuming the first result is what you want

    # Convert the model label to a more readable form
    converted_label = convert_label(sentiment_result['label'])
    print(f"Sentiment result: {sentiment_result}")  # Debugging print before conversion
    print(f"Converted sentiment: {converted_label}")  # After conversion

    # Returning both the converted label and confidence score
    return {'label': converted_label, 'score': sentiment_result['score']}




@app.route('/')
def index():
    return render_template('index.html', fields=fields)


@app.route('/analyze', methods=['POST'])
def analyze():
    analysis_type = request.form.get('analysis_type')

    if analysis_type == 'field_topic':
        selected_field = request.form.get('field')
        user_input = request.form.get('topic')

        hindi_words_df = pd.read_csv("romanhindi_data.csv")
        hindi_words_set = set(hindi_words_df['Word'].str.lower())

        if selected_field == "Entertainment":
            subreddits_to_analyze = ['Bollywood', 'IndianCinema', 'IndiaSpeaks']
        elif selected_field == "Technology":
            subreddits_to_analyze = ['IndiaTech', 'IndianGaming', 'technology', 'technews']
        elif selected_field == 'Politics':
            subreddits_to_analyze = ['indiaPolitics', 'IndiaSpeaks', 'indianews', 'India']
        else:
            return "Invalid field"

        comments_df = collect_comments(user_input, subreddits_to_analyze)

        translator = Translator()

        preprocessed_comments_df = preprocess_comments(comments_df)
        hindi_comments_df, english_comments_df = separate_hindi_english(preprocessed_comments_df)

        if not hindi_comments_df.empty:
            preprocessed_hindi_comments_df = preprocess_comments(hindi_comments_df)
            # Apply sentiment analysis on Pure Hindi comments
            hindi_with_sentiment_df = analyze_sentiment_hindi(preprocessed_hindi_comments_df)
            hindi_sentiment_output_path = f"{user_input}_hindi_sentiment.csv"
            hindi_with_sentiment_df.to_csv(hindi_sentiment_output_path, index=False)
            print(f"Sentiment analysis for pure Hindi comments saved to {hindi_sentiment_output_path}")
            # Add a 'source' column for plotting later
            hindi_with_sentiment_df['source'] = 'Pure Hindi'
        else:
            print("No Hindi data available for this topic.")

        roman_hindi_df, pure_english_df = separate_roman_hindi(english_comments_df, hindi_words_set)
        roman_hindi_df['language'] = 'hi'
        translated_roman_hindi_df = translate_comments(roman_hindi_df, translator)

        # Apply sentiment analysis on Translated roman hindi comments
        translated_roman_hindi_with_sentiment_df = analyze_sentiment_english(translated_roman_hindi_df)
        roman_hindi_sentiment_output_path = f"{user_input}_roman_hindi_sentiment.csv"
        translated_roman_hindi_with_sentiment_df.to_csv(roman_hindi_sentiment_output_path, index=False)
        print(f"Sentiment analysis for Translated Roman Hindi data saved to {roman_hindi_sentiment_output_path}")

        # Apply sentiment analysis on Pure English comments
        pure_english_with_sentiment_df = analyze_sentiment_english(pure_english_df)
        pure_english_sentiment_output_path = f"{user_input}_pure_english_sentiment.csv"
        pure_english_with_sentiment_df.to_csv(pure_english_sentiment_output_path, index=False)
        print(f"Sentiment analysis for Pure English comments saved to {pure_english_sentiment_output_path}")

        # Add a 'source' column to each DataFrame before concatenation
        # Initialize an empty list to hold DataFrames that are not empty
        dataframes_to_combine = []

        # Append DataFrames to the list if they are not empty and have been processed
        if not hindi_comments_df.empty:
            hindi_with_sentiment_df['source'] = 'Pure Hindi'
            save_plot(hindi_with_sentiment_df, 'Sentiment Distribution for Pure Hindi Comments', 'hindi_sentiment_plot.png')
            dataframes_to_combine.append(hindi_with_sentiment_df)

        if not roman_hindi_df.empty:
            translated_roman_hindi_with_sentiment_df['source'] = 'Translated Roman Hindi'
            save_plot(translated_roman_hindi_with_sentiment_df, 'Sentiment Distribution for Translated Roman Hindi Comments', 'translated_roman_hindi_sentiment_plot.png')
            dataframes_to_combine.append(translated_roman_hindi_with_sentiment_df)

        if not pure_english_df.empty:
            pure_english_with_sentiment_df['source'] = 'Pure English'
            save_plot(pure_english_with_sentiment_df, 'Sentiment Distribution for Pure English Comments', 'pure_english_sentiment_plot.png')
            dataframes_to_combine.append(pure_english_with_sentiment_df)

        if dataframes_to_combine:
            combined_sentiment_counts, sentiment_summary = aggregate_sentiments(dataframes_to_combine)
            plot_combined_sentiment_distribution(combined_sentiment_counts, 'Percentage Distribution of Sentiments Across Datasets', 'combined_sentiment_plot.png')
        
        result = f"Sentiment analysis result for {user_input} in {selected_field} (Analysis Type: {analysis_type})"
        return render_template('results_field_topic.html',
                           result=result,
                           hindi_data_available=not hindi_comments_df.empty,
                           roman_hindi_data_available=not roman_hindi_df.empty,
                           pure_english_data_available=not pure_english_df.empty, sentiment_summary=sentiment_summary)

    elif analysis_type == 'custom_text':
        user_input = request.form.get('custom_text')
        sentiment_result = analyze_sentiment(user_input)
        result = {
            'text': user_input,
            'sentiment': sentiment_result['label'],  
            'confidence': sentiment_result['score']  
        }
        return render_template('results_custom_text.html', result=result)

    return "Invalid analysis type"


if __name__ == '__main__':
    app.run(debug=True)