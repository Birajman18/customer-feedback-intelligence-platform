# Customer Feedback Intelligence Platform: A Data-Driven Review Analytics System Utilizing Machine Learning, LLM-Based NLP, and Statistical Analysis

An end-to-end platform that analyzes customer reviews using machine learning for sentiment classification and a local LLM for theme extraction, delivering actionable business insights through an interactive Streamlit dashboard.

---

## Table of Contents

- [Demo](#demo)
- [Problem Statement](#problem-statement)
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Pipeline Flow](#pipeline-flow)
- [Sentiment Classification Model](#sentiment-classification-model)
- [Theme Extraction (LLM)](#theme-extraction-llm)
- [Dashboard Features](#dashboard-features)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Output Files](#output-files)
- [Future Work](#future-work)

---

## Demo

[Watch the full demo video](https://drive.google.com/file/d/10XaJrL9f8GI3zWYUAqgo1Yq-ICwLuyYh/view?usp=sharing)

---

## Problem Statement

Businesses receive thousands of customer reviews across platforms like Yelp, Google Reviews, and social media. Manually reading and categorizing these reviews to identify trends, recurring complaints, and areas of improvement is impractical at scale.

Without an automated system, companies face several challenges:

- **Volume** — A single location can accumulate hundreds of reviews per month. Multi-location businesses deal with thousands. No human team can read them all consistently.
- **Delayed response** — By the time a manager manually identifies a spike in complaints about service speed or product quality, the damage to customer retention is already done.
- **Missed patterns** — Individual reviews may seem like isolated incidents, but automated analysis can reveal that 40% of all negative reviews mention the same theme, pointing to a systemic issue.
- **No historical tracking** — Without structured data, businesses cannot compare how customer sentiment around specific topics has changed over months or years.

This platform solves these problems by automating the entire pipeline — from raw review text to structured, filterable, actionable insights — in minutes instead of weeks.

---

## Overview

The platform takes raw customer review data (CSV format) and runs it through a two-stage AI pipeline:

1. **Sentiment Classification** — A Logistic Regression model trained on TF-IDF features predicts whether each review is positive, negative, or neutral.
2. **Theme Extraction** — A locally running LLM (Gemma 2 9B via Ollama) reads each review and tags it with one or more business themes from a fixed list of 8 categories.

The results are then visualized in an interactive Streamlit dashboard that provides sentiment distribution, theme analysis, spike detection, time-based trends, and the ability to read actual customer reviews filtered by theme and sentiment.

---

## Architecture

```
Raw CSV Reviews
      |
      v
[Preprocessing] --- clean text, normalize, lowercase
      |
      v
[ML Model] --- TF-IDF vectorization + Logistic Regression
      |         outputs: predicted_sentiment, confidence, is_mixed
      v
[LLM Pipeline] --- Gemma 2 9B via Ollama (batched, retry logic)
      |             outputs: themes per review (from 8 fixed categories)
      v
[Streamlit Dashboard] --- interactive visualizations, PDF export
```

---

## Project Structure

```
Customer-Feedback-Intelligence-Platform/
|
|-- .streamlit/
|   |-- config.toml                  # Streamlit server config (upload size limit)
|
|-- output/
|   |-- sentiment_model.pkl          # Trained sentiment classification model
|   |-- tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
|   |-- predicted_data_ml.csv        # Model predictions on test set
|   |-- vocab_data_ml.csv            # Feature importance data
|   |-- confusion_matrix_ml.png      # Model evaluation visualization
|   |-- log_file_ml.log              # Training metrics and logs
|
|-- src/
|   |-- resources/
|   |   |-- saved_tabs.json          # Saved dashboard layout presets
|   |
|   |-- sample_data/
|   |   |-- training_testing_data.csv         # Primary training dataset
|   |   |-- starbucks_reviews_sample_15000.csv # Starbucks review data
|   |   |-- amazon_grocery_sample_15000.csv    # Amazon grocery review data
|   |   |-- sampled_reviews_3000.csv           # Smaller sample for quick testing
|   |   |-- test_data_additional.csv           # Additional test data
|   |
|   |-- scripts/
|       |-- main.py                  # Entry point, orchestrates the pipeline
|       |-- pipeline_ml.py           # ML model training, prediction, mixed rule
|       |-- pipeline_llm.py          # LLM theme extraction with retry logic
|       |-- pipeline_ui.py           # Streamlit dashboard rendering
|
|-- .gitignore
|-- requirements.txt                 # Python dependencies
|-- Start_Dashboard.bat              # Windows launcher script
|-- README.md
```

---

## Pipeline Flow

The pipeline executes in this order when a user uploads a CSV and clicks "Run Analysis":

**Step 1 — Model Loading**

The app checks if `sentiment_model.pkl` and `tfidf_vectorizer.pkl` exist in the output folder. If they do, it loads them instantly. If not, it trains the model from `training_testing_data.csv` and saves the pkl files for future use. Training only happens once.

**Step 2 — Preprocessing**

The uploaded CSV is scanned for a text column (`text`, `raw_text`, or `clean_text`). The text is lowercased, missing values are filled, and a standardized `clean_text` column is created.

**Step 3 — Sentiment Prediction**

Each review is converted to a TF-IDF vector (up to 5,000 features, unigrams and bigrams) and passed through the trained Logistic Regression model. The model outputs a predicted sentiment label (positive, negative, or neutral), a confidence score (the highest class probability), and individual probabilities for each class. Reviews predicted as neutral are further checked by a rule-based mixed sentiment detector that looks for contrast words ("but", "however") and dual polarity vocabulary (both positive and negative cue words in the same review).

**Step 4 — Theme Extraction**

Reviews are sent in batches of 30 to a locally running Gemma 2 9B model via Ollama. Each batch is wrapped in a structured prompt that instructs the LLM to assign one or more themes from a fixed list of 8 categories. The LLM response is parsed as JSON, validated against the allowed theme list, and retried up to 5 times if the response is malformed or contains hallucinated themes. Two batches are processed in parallel using ThreadPoolExecutor.

**Step 5 — Dashboard Rendering**

The enriched dataframe (with sentiment, confidence, is_mixed, and themes columns) is passed to the Streamlit dashboard for interactive visualization.

---

## Sentiment Classification Model

| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF (max 5,000 features) |
| N-gram range | (1, 2) — unigrams and bigrams |
| Train/Test split | 80% / 20% (stratified) |
| Classes | Positive, Negative, Neutral |
| Regularization | C=0.8 |

**Mixed Sentiment Detection**

Reviews classified as neutral are passed through an additional rule-based check. A review is flagged as `is_mixed = True` if:
- The model assigns similar probabilities to both positive and negative classes (both above 30%, difference under 25%) AND the text contains a contrast word, OR
- The text contains both a contrast word and dual polarity vocabulary (positive + negative cue words together).

---

## Theme Extraction (LLM)

The platform uses Gemma 2 9B running locally through Ollama. Each review is classified into one or more of these 8 business themes:

| Theme | Description |
|-------|-------------|
| Product Quality | Food, drink, or item quality and standards |
| Product Availability | Out of stock items or limited menu |
| Customer Service | Staff attitude, helpfulness, complaint handling |
| Speed of Service | Wait times, queue length, order delays |
| Store Environment | Cleanliness, atmosphere, seating, parking |
| Price and Value | Cost, affordability, value for money |
| Digital and Rewards | App, online ordering, loyalty points |
| Policies and Safety | Return policies, hygiene, health precautions |

The extraction pipeline includes hallucination prevention — any theme returned by the LLM that does not match the approved list is rejected and the batch is retried. Failed batches (after 5 retries or 3-minute timeout) are dropped from the final dataset.

---

## Dashboard Features

The Streamlit dashboard provides the following visualization modules, each of which can be toggled on/off and rearranged by the user:

**Prediction Summary** — KPI cards showing total reviews, average model confidence, and mixed sentiment rate. Includes a sentiment distribution pie chart and a confidence box plot showing prediction uncertainty across classes.

**Overall Sentiment Over Time** — Monthly line chart tracking positive, negative, and neutral review volume over time.

**Top Extracted Themes** — Horizontal bar chart and data table showing the most frequently mentioned themes across all reviews.

**Theme Sentiment Distribution** — Select any theme from a dropdown and see its sentiment breakdown as a pie chart, with a detailed comparison table showing counts and percentages against the total dataset.

**Time-Oriented Trends** — Line chart showing how each theme's mention volume changes month over month.

**Phrases and Reviews** — Interactive review explorer with two tabs. The "Top AI Matches" tab uses the LLM to analyze reviews for a selected theme and sentiment, extracting actionable insights and citing specific reviews as evidence. The "Random Feed" tab shows randomly sampled reviews with a refresh button.

**Negative Spike Detection** — Bar chart of monthly negative review counts with a statistical threshold line (mean + 1.5 standard deviations). Months exceeding the threshold are highlighted and listed as detected spikes.

**Theme Sentiment Heatmap** — Color-coded grid showing the percentage distribution of each sentiment across all themes. Darker cells indicate higher concentration.

**Additional Features:**
- Customizable dashboard layout with save/load presets
- Date range filter for time-based analysis
- 1/2/3 column grid layout options
- PDF report export with optional company logo
- About and Methodology section

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Machine Learning | scikit-learn, Logistic Regression, TF-IDF |
| LLM | Ollama, Gemma 2 9B |
| Dashboard | Streamlit, Plotly |
| Data Processing | pandas, NumPy |
| PDF Export | fpdf2, kaleido |
| Model Persistence | joblib |

---

## Setup and Installation

**Prerequisites:**
- Python 3.10 or higher
- Ollama installed with Gemma 2 9B model pulled

**Step 1 — Clone the repository**

```bash
git clone https://github.com/your-username/Customer-Feedback-Intelligence-Platform.git
cd Customer-Feedback-Intelligence-Platform
```

**Step 2 — Create and activate a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Install and start Ollama**

Download Ollama from [ollama.com](https://ollama.com) and pull the model:

```bash
ollama pull gemma2:9b
```

**Step 5 — Run the dashboard**

```bash
streamlit run src/scripts/main.py
```

Or on Windows, double-click `Start_Dashboard.bat`.

The dashboard will open in your browser at `http://localhost:8501`.

---

## Usage

1. Open the dashboard in your browser.
2. Upload a CSV file containing customer reviews. The CSV must have a text column named `text`, `raw_text`, or `clean_text`.
3. Click "Run Analysis" in the sidebar.
4. The pipeline will preprocess the reviews, predict sentiment, and extract themes. Theme extraction progress is shown via a progress bar.
5. Once complete, the dashboard displays all visualizations. Use the sidebar to customize which modules are visible, change the grid layout, filter by date range, or export a PDF report.

---

## Dataset

The training data was extracted from the Yelp Open Dataset, filtered to Starbucks reviews. Star ratings were converted to sentiment labels:

| Stars | Sentiment |
|-------|-----------|
| 4-5 | Positive |
| 3 | Neutral |
| 1-2 | Negative |

Additional test datasets include Amazon grocery reviews and supplementary Starbucks samples for cross-domain validation.

---

## Output Files

The `output/` folder contains artifacts generated during model training and evaluation:

| File | Description |
|------|-------------|
| `sentiment_model.pkl` | Trained Logistic Regression model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `predicted_data_ml.csv` | Predictions on the test set with confidence scores |
| `vocab_data_ml.csv` | Feature importance and sentiment coefficients per word |
| `confusion_matrix_ml.png` | Confusion matrix visualization |
| `log_file_ml.log` | Training metrics (accuracy, F1, log loss) |

---

## Future Work

**Cloud-based LLM Integration** — The current pipeline uses a locally hosted LLM (Gemma 2 9B via Ollama) for theme extraction. While this ensures data privacy and zero API costs, it requires significant local compute resources and processing time. A natural next step would be integrating cloud-based models like OpenAI GPT-4, Anthropic Claude, or Google Gemini through their APIs. This would dramatically reduce theme extraction time from minutes to seconds, improve classification accuracy due to stronger model capabilities, and remove the dependency on local GPU hardware. The existing `call_llm()` function is designed to be swappable — replacing the Ollama call with an API call requires minimal code changes.

**Real-time Data Ingestion** — Currently the platform processes static CSV uploads. Integrating with review platform APIs (Yelp, Google Reviews, social media) would allow continuous monitoring with automated alerts when negative sentiment spikes or new complaint themes emerge.

**Multi-language Support** — The current model and LLM pipeline are English-only. Adding multilingual support through translation layers or multilingual models would expand the platform's applicability to global businesses.

**Advanced ML Models** — The Logistic Regression classifier could be upgraded to transformer-based models like BERT or DistilBERT for improved sentiment classification accuracy, particularly on nuanced or sarcastic reviews that TF-IDF-based models struggle with.

---

## Author

Birajman Tamang — [GitHub](https://github.com/Birajman18)