import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import threading
import time
import json
import keyboard

# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Create a set of stopwords for text preprocessing
stop_words = set(stopwords.words('english'))  # Change 'english' to the appropriate language

# Function to load inappropriate words from a language-specific file
def load_inappropriate_words(language):
    with open(f'inappropriate_words_{language}.txt', 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

# Function to load language codes mapping from a configuration file
def load_language_codes():
    with open('language_codes.json', 'r') as file:
        return json.load(file)

# Define a mapping of supported languages to their respective codes
language_codes = load_language_codes()

# Function to analyze text for inappropriate content and sentiment
def analyze_text(text, language):
    # Preprocess the text
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]

    # Load language-specific inappropriate words
    inappropriate_words = load_inappropriate_words(language)

    # Check for inappropriate content
    contains_inappropriate_content = any(word in inappropriate_words for word in words)

    # Analyze sentiment
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    return contains_inappropriate_content, compound_score

# Function to monitor keyboard events and analyze typed text
def monitor_typing():
    global input_text
    input_text = ""
    while True:
        time.sleep(1)  # Adjust the interval as needed
        typed_text = keyboard.read_event(suppress=True).name
        if typed_text:
            input_text += typed_text
            contains_inappropriate, sentiment = analyze_text(input_text, "en")  # Change the language code as needed
            if contains_inappropriate:
                print("Inappropriate content detected!")
                # Take appropriate actions here (e.g., warn the user, block content)
            if sentiment >= 0.05:
                print("Sentiment: Positive")
            elif sentiment <= -0.05:
                print("Sentiment: Negative")
            else:
                print("Sentiment: Neutral")

# Create a separate thread to monitor typing
typing_thread = threading.Thread(target=monitor_typing)
typing_thread.daemon = True
typing_thread.start()

# Keep the program running
while True:
    time.sleep(1)
