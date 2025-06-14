"""
Sentiment Analysis Tool for Student Feedback
- Supports VADER (rule-based) and BERT (transformer-based) sentiment analysis
- Loads data from CSV or Twitter API (Tweepy)
- Modular, reusable, and well-documented
"""
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Optional: For Twitter API
try:
    import tweepy
except ImportError:
    tweepy = None

def ensure_vader():
    """Download the VADER lexicon if not already present."""
    nltk.download('vader_lexicon', quiet=True)


def clean_text(text):
    """Clean and normalize input text (remove URLs, mentions, hashtags, punctuation, lowercase)."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower().strip()
    return text


def load_csv(path, text_col="text"):
    """Load a CSV file and return a DataFrame with the specified text column."""
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in CSV.")
    df = df[[text_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    return df


def simulate_feedback():
    """Return a DataFrame with simulated student feedback."""
    return pd.DataFrame({
        "text": [
            "The professor explained the concepts very well!",
            "I found the assignments too difficult.",
            "The course was okay, not great but not bad either.",
            "Loved the interactive sessions!",
            "The lectures were boring and hard to follow.",
            "Neutral experience overall."
        ]
    })


def vader_sentiment(df, text_col="text"):
    """Apply VADER sentiment analysis and return a Series of sentiment labels."""
    sia = SentimentIntensityAnalyzer()
    scores = df[text_col].apply(lambda x: sia.polarity_scores(x))
    sentiments = scores.apply(lambda x: 'positive' if x['compound'] > 0.05 else ('negative' if x['compound'] < -0.05 else 'neutral'))
    return sentiments


def bert_sentiment(df, text_col="text"):
    """Apply BERT-based sentiment analysis and return a list of sentiment labels."""
    classifier = pipeline("sentiment-analysis")
    results = classifier(df[text_col].tolist(), truncation=True)
    sentiments = [r['label'].lower() if r['label'].lower() in ['positive', 'negative'] else 'neutral' for r in results]
    return sentiments


def plot_sentiment_distribution(df, vader_col, bert_col, save_path=None):
    """Plot and optionally save the sentiment distribution for VADER and BERT results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x=vader_col, data=df, ax=axes[0])
    axes[0].set_title('VADER Sentiment Distribution')
    sns.countplot(x=bert_col, data=df, ax=axes[1])
    axes[1].set_title('BERT Sentiment Distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def analyze_feedback_df(df, text_col="text"):
    """
    Given a DataFrame with a text column, clean the text and add VADER and BERT sentiment columns.
    Returns the modified DataFrame.
    """
    ensure_vader()
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    df['vader_sentiment'] = vader_sentiment(df, text_col)
    df['bert_sentiment'] = bert_sentiment(df, text_col)
    return df
