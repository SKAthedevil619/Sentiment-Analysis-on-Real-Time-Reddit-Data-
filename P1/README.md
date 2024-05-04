# Real-Time Sentiment Analysis on Reddit Data

This project aims to perform sentiment analysis on real-time Reddit comments, focusing on various topics across different subreddits. By analyzing the sentiment of user comments, we can gain insights into public opinion and emotional trends related to specific subjects, such as entertainment, technology, and politics.

## Overview

The project utilizes Python libraries such as praw, transformers, nltk, and pandas to collect, preprocess, and analyze Reddit comments. It employs pre-trained sentiment analysis models to classify comments into positive, negative, or neutral sentiments. The sentiment analysis pipeline includes:

1. *Data Collection*: Reddit comments are collected in real-time using the praw library, focusing on relevant subreddits related to the chosen topic.

2. *Preprocessing*: Comment text is preprocessed to remove noise, including URLs, user mentions, special characters, and stopwords. Text normalization techniques such as tokenization, lemmatization, and lowercasing are applied.

3. *Sentiment Analysis*: The preprocessed text is passed through pre-trained sentiment analysis models, such as BERT or RoBERTa, to predict sentiment labels for each comment.

4. *Visualization*: Sentiment distributions across different datasets (e.g., pure Hindi, translated Roman Hindi, pure English) are visualized using matplotlib and seaborn, providing insights into sentiment trends.

5. *Evaluation*: The accuracy of sentiment predictions is evaluated using sklearn's accuracy_score metric, comparing model predictions with ground truth sentiment labels if available.

## Usage

1. Clone the repository:


2. Install dependencies:


3. Run the Flask application:


4. Access the application in your web browser at http://localhost:5000 to perform sentiment analysis on custom text or selected Reddit topics.

## Dependencies

- praw: Python Reddit API Wrapper for accessing Reddit data.
- transformers: Hugging Face's library for state-of-the-art Natural Language Processing (NLP) models.
- nltk: Natural Language Toolkit for text preprocessing tasks.
- pandas: Data manipulation library for handling tabular data.
- scikit-learn: Machine learning library for model evaluation and metrics.
## Conclusion

Real-time sentiment analysis on Reddit data offers valuable insights into public opinions and emotional trends across various topics and communities. Through this project, we have demonstrated the following key findings and outcomes:

- *Insightful Analysis*: By analyzing Reddit comments, we were able to gain insights into sentiment trends related to different topics, including entertainment, technology, and politics. This analysis provided valuable information about public perception and sentiment on these subjects.

- *Multilingual Support*: The project supports sentiment analysis in multiple languages, including English and Hindi. By leveraging translation services and language detection algorithms, we were able to analyze sentiments in comments written in different languages.

- *Model Evaluation*: We evaluated the performance of sentiment analysis models using accuracy metrics. This allowed us to assess the reliability and accuracy of sentiment predictions and identify areas for improvement.

- *Visualization*: Visualizing sentiment distributions using matplotlib and seaborn helped in presenting the results in an intuitive and easy-to-understand manner. These visualizations provided a comprehensive overview of sentiment trends across different datasets.

- *Interactive Web Interface*: The project includes a Flask-based web application that allows users to perform sentiment analysis on custom text or selected Reddit topics. This interactive interface enhances accessibility and usability, enabling users to explore sentiment analysis results effortlessly.

In conclusion, real-time sentiment analysis on Reddit data is a powerful tool for understanding public sentiment and emotional trends in online communities. This project demonstrates the potential of NLP techniques and machine learning models in extracting valuable insights from user-generated content on social media platforms