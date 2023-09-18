import threading
import time
import keyboard
import matplotlib.pyplot as plt
from transformers import pipeline

# Function to monitor keyboard events and analyze typed text
def monitor_typing():
    global input_text, x_values, y_values, sentiment_analyzer
    input_text = ""
    buffer = ""
    while True:
        typed_text = keyboard.read_event(suppress=True).name
        if typed_text:
            input_text += typed_text
            buffer += typed_text
            if len(buffer) >= 10:
                sentiment = sentiment_analyzer.analyze_sentiment(input_text)
                if sentiment == "POSITIVE":
                    print("Sentiment: Positive")
                elif sentiment == "NEGATIVE":
                    print("Sentiment: Negative")
                else:
                    print("Sentiment: Neutral")
                buffer = ""
                x_values.append(time.time())
                y_values.append(1 if sentiment == "POSITIVE" else (-1 if sentiment == "NEGATIVE" else 0))

# Function to update the real-time sentiment trend plot
def update_graph(timer):
    global x_values, y_values
    if len(x_values) > max_data_points:
        x_values.pop(0)
        y_values.pop(0)
    plt.clf()
    plt.plot(x_values, y_values)
    plt.xlabel("Time")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Trend Over Time")

max_data_points = 100
x_values = []
y_values = []

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")

    def analyze_sentiment(self, text):
        result = self.classifier(text)
        label = result[0]['label']
        return label

sentiment_analyzer = SentimentAnalyzer()

# Create a timer to update the graph every second
graph_timer = threading.Timer(1, update_graph, [None])
graph_timer.start()

# Create a separate thread to monitor typing
typing_thread = threading.Thread(target=monitor_typing)
typing_thread.daemon = True
typing_thread.start()

# Keep the program running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    graph_timer.cancel()  # Cancel the timer when exiting
    plt.ioff()  # Deactivate interactive mode
    plt.show()  # Keep the plot window open
