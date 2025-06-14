# Sentiment Analysis of Campus Feedback

A web-based tool for analyzing student feedback (course evaluations or social media posts) using both rule-based (VADER) and transformer-based (BERT) models. Easily upload CSVs or enter text, visualize sentiment, and export results.

---

## Features
- **Upload CSV** or enter feedback manually
- **Sentiment analysis** with VADER (NLTK) and BERT (Hugging Face Transformers)
- **Interactive web interface** (Flask + Bootstrap)
- **Visualize results** with sentiment distribution charts
- **Download labeled results** as CSV
- **Modular, extensible code**

---

## Demo
![Sentiment Chart Example](static/sentiment_chart.png)

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/campus-sentiment-analysis.git
cd campus-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the web app
```bash
python app.py
```
Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## Usage
- **Upload** a CSV file with a `text` column (each row is feedback)
- **Or** enter feedback manually in the text box
- **View** sentiment results and distribution charts
- **Download** the full results as CSV

---

## Technologies Used
- Python 3
- Flask
- Pandas, Matplotlib, Seaborn
- NLTK (VADER)
- Transformers (BERT)
- Bootstrap 5
- Tweepy (for optional Twitter API integration)

---

## Extending (Optional)
- **Twitter Analysis**: Integrate the Twitter API via Tweepy to fetch and analyze tweets
- **Advanced Visuals**: Add time trends, word clouds, or more charts
- **API Keys**: Store Twitter API keys securely (if using Tweepy)

---

## Project Structure
```
Sentiment Analysis/
├── app.py                 # Flask web app
├── sentiment_analysis.py  # Core sentiment analysis functions
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Main web interface
├── static/
│   └── sentiment_chart.png# Chart images
├── uploads/               # Uploaded CSVs
├── results/               # Output CSVs and charts
└── README.md
```

---

## Contact
For questions or contributions, open an issue or PR on GitHub.
