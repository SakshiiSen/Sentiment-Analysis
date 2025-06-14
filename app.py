"""
Flask web app for Sentiment Analysis of Student Feedback
"""
import os
import io
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from sentiment_analysis import analyze_feedback_df, plot_sentiment_distribution

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle CSV upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                df = pd.read_csv(filepath)
                analyzed_df = analyze_feedback_df(df)
                result_csv = os.path.join(app.config['RESULT_FOLDER'], 'sentiment_results.csv')
                analyzed_df.to_csv(result_csv, index=False)
                # Plot and save chart
                chart_path = os.path.join(app.config['RESULT_FOLDER'], 'sentiment_chart.png')
                plot_sentiment_distribution(analyzed_df, 'vader_sentiment', 'bert_sentiment', save_path=chart_path)
                # Prepare data for interactive rendering
                rows = analyzed_df.head(10).to_dict(orient='records')
                vader_counts = analyzed_df['vader_sentiment'].value_counts().to_dict()
                bert_counts = analyzed_df['bert_sentiment'].value_counts().to_dict()
                return render_template(
                    'index.html',
                    results=rows,
                    vader_counts=vader_counts,
                    bert_counts=bert_counts,
                    chart_url=url_for('static', filename='sentiment_chart.png'),
                    download_url=url_for('download_csv')
                )
            else:
                flash('Invalid file type. Please upload a CSV file.')
        # Handle manual text input
        elif 'feedback' in request.form and request.form['feedback'].strip():
            feedback = request.form['feedback'].strip()
            df = pd.DataFrame({'text': [feedback]})
            analyzed_df = analyze_feedback_df(df)
            chart_path = os.path.join(app.config['RESULT_FOLDER'], 'sentiment_chart.png')
            plot_sentiment_distribution(analyzed_df, 'vader_sentiment', 'bert_sentiment', save_path=chart_path)
            rows = analyzed_df.to_dict(orient='records')
            vader_counts = analyzed_df['vader_sentiment'].value_counts().to_dict()
            bert_counts = analyzed_df['bert_sentiment'].value_counts().to_dict()
            return render_template(
                'index.html',
                results=rows,
                vader_counts=vader_counts,
                bert_counts=bert_counts,
                chart_url=url_for('static', filename='sentiment_chart.png'),
                download_url=None
            )

        else:
            flash('Please upload a CSV file or enter feedback text.')
    return render_template('index.html', results=None, vader_counts=None, bert_counts=None, chart_url=None, download_url=None)

@app.route('/download')
def download_csv():
    result_csv = os.path.join(app.config['RESULT_FOLDER'], 'sentiment_results.csv')
    if os.path.exists(result_csv):
        return send_file(result_csv, mimetype='text/csv', as_attachment=True, download_name='sentiment_results.csv')
    else:
        flash('No results available for download.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
