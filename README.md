# Customer Experience Analytics for Ethiopian Banking Apps

This project simulates the role of a data analyst at Omega Consultancy, where mobile banking app reviews for three major Ethiopian banks—Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank—are to be analyzed.

## Project Challenge Overview
As part of a consulting engagement, user reviews from the Google Play Store are to be scraped and analyzed to help these banks identify satisfaction drivers, recurring complaints, and areas for improvement.

### Business Objective
Support is being provided to improve customer retention and satisfaction through the following steps:
- Reviews are to be scraped from the Google Play Store.
- Sentiments (positive, neutral, negative) are to be analyzed and common themes extracted.
- Satisfaction drivers (e.g., speed, design) and pain points (e.g., crashes, login errors) are to be identified.
- Cleaned review data is to be stored in an Oracle database.
- Actionable recommendations and visual insights are to be delivered.

### Scenarios Simulated
1. **User Retention**: Complaints about slow loading are to be investigated to determine whether the issue is systemic.
2. **Feature Enhancement**: Features demanded by users (e.g., fingerprint login) are to be extracted and analyzed.
3. **Complaint Management**: Complaint clusters such as “login error” are to be tracked to support chatbot integration and enhance support resolution.

## Tasks & Deliverables
### Task 1: Data Collection and Preprocessing
- At least 400 reviews per bank are to be scraped using `google-play-scraper`.
- Reviews are to be preprocessed: duplicates removed, missing data handled, and dates normalized.
- The cleaned data is to be saved as CSV.

### Task 2: Sentiment and Thematic Analysis
- Sentiments are to be classified using models like DistilBERT or VADER.
- Keywords are to be extracted using TF-IDF or spaCy.
- Keywords are to be grouped into 3–5 themes per bank (e.g., "UI/UX", "Crashes", "Slow Transfers").
- The enriched dataset with sentiment and theme labels is to be saved.

### Task 3: Oracle Database Integration
- A relational schema with tables for banks and reviews is to be designed.
- The cleaned data is to be inserted using Python + `cx_Oracle`.
- The SQL schema and connection script are to be committed.

### Task 4: Insights and Recommendations
- At least 2 satisfaction drivers and 2 pain points per bank are to be identified.
- Visualizations (e.g., sentiment bar charts, keyword clouds) are to be created.
- Actionable recommendations for app improvement are to be provided.

## Folder Structure
- `scripts/`: Modular scripts are to be placed here for scraping, preprocessing, NLP, database interaction, and plotting.
- `tests/`: Unit tests with Pytest are to be stored for code reliability.
- `data/`: Raw and processed datasets are to be saved.
- `notebooks/`: Exploratory and final analysis notebooks are to be stored.

## Setup Instructions
```bash
pip install -r requirements.txt
python run.py     # Scraping and preprocessing are executed
pytest tests/     # Unit tests are executed
```

## Git Branching Strategy
- `main`: Stable version
- `task-1`: Scraping and preprocessing
- `task-2`: Sentiment and theme extraction
- `task-3`: Oracle DB integration
- `task-4`: Visualization and reporting

## KPIs
- Over 1200 clean reviews (400 per bank) are to be collected
- Sentiment scores should be generated for at least 90% of reviews
- At least 3 themes per bank are to be identified
- A minimum of 2 drivers and 2 pain points per bank are to be included
- 3–5 insightful and clearly labeled visualizations are to be generated
- Code is to be modular, tested, and version-controlled
