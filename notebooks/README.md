
# 📝 Notebooks – Customer Experience Analytics for Banking Apps

This directory hosts the core Jupyter notebooks used to explore, process, and analyze customer experience data from banking applications.

## 📂 Contents

- `preprocessing.ipynb`  
  - Data ingestion, cleaning, normalization, and feature engineering.  
  - Preps raw customer feedback and usage data for analysis.

- `scrape.ipynb`  
  - Web-scraping logic for collecting app reviews/feedback from Google Play Store and/or Apple App Store.  
  - Includes URL generation, HTML parsing, error handling, and output formatting for downstream use.

- `*.ipynb` (hidden in screenshot)  
  - Name partially obscured—likely focused on sentiment analysis, exploratory data analysis, or modeling.  
  - Please explore this file to confirm its purpose and give it a descriptive name (e.g., `eda.ipynb`, `sentiment_analysis.ipynb`, `modeling.ipynb`).

## ⚙️ Setup & Dependencies

Make sure your environment has the needed libraries:

```bash
pip install -r ../requirements.txt
```

Typically used packages include:
- `pandas`, `numpy`
- `requests`, `BeautifulSoup`, or `Selenium`
- `nltk`, `spacy`, `scikit-learn`
- `matplotlib`, `seaborn`

## 🚀 How to Use

1. Clone the full repo and navigate to `notebooks/`:
   ```bash
   git clone https://github.com/EstiphanosH/Customer-Experience-Analytics-for-Banking-apps.git
   cd Customer-Experience-Analytics-for-Banking-apps/notebooks
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open `preprocessing.ipynb` first, then `scrape.ipynb`, followed by the remaining notebook(s) in logical sequence.

## 🔄 Recommended Notebook Execution Order

1. **Data acquisition** (`scrape.ipynb`) – to collect raw review data  
2. **Data preparation** (`preprocessing.ipynb`) – to clean and structure data  
3. **Analysis/modeling** (`<hidden-notebook>.ipynb`) – for EDA, sentiment analysis, clustering or churn modeling

## 📝 Notebook Descriptions

### `preprocessing.ipynb`
- **Purpose**: Cleans and prepares raw review data for analysis.
- **Key Steps**:
  - Data ingestion from raw CSV files.
  - Handling missing values, duplicate entries, and outliers.
  - Text preprocessing (lowercasing, removing punctuation, etc.).
  - Feature engineering (e.g., creating a `review_date_normalized` column).
  - Saving cleaned data to CSV for further analysis.


### `scrape.ipynb`
- **Purpose**: Collects raw review data from Google Play Store.
- **Key Steps**:
  - Defines app IDs and bank name mapping.
  - Sets output directory and date for saving scraped data.
  - Scrapes reviews for all defined apps.
  - Saves scraped data to CSV files.
  - Displays scraping results for verification.

### `sentiment_analysis.ipynb`
- **Purpose**: Performs sentiment analysis on cleaned review data.
- **Key Steps**:
  - Loads cleaned review data from CSV.
  - Applies sentiment analysis using a pre-trained model (e.g., DistilBERT).
  - Adds sentiment labels and scores to the DataFrame.
  - Saves the DataFrame with sentiment analysis results to CSV.

### `exploratory_data_analysis.ipynb`
- **Purpose**: Conducts exploratory data analysis (EDA) on cleaned and analyzed review data.
- **Key Steps**:
  - Loads data from CSV.
  - Generates summary statistics and visualizations.
  - Identifies key themes and sentiment distributions.
  - Saves EDA results and visualizations for reporting.

