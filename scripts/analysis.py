import pandas as pd
import os
from tqdm import tqdm
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import logging
from collections import Counter
import json

from utils import ANALYSIS_DATA_DIR, FIGURES_DIR, TODAY_DATE_STR, THEME_KEYWORDS, download_spacy_model, remove_non_printable # Import THEME_KEYWORDS and remove_non_printable

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize sentiment pipeline (DistilBERT)
sentiment_pipeline = None
SENTIMENT_MODEL_READY = False
print("Loading sentiment analysis model...")
try:
    # Explicitly set device to 'cpu' if not using GPU, can prevent issues
    # import torch
    # device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english") #, device=device)
    print("Sentiment analysis model loaded.")
    SENTIMENT_MODEL_READY = True
except Exception as e:
    print(f"Error loading sentiment analysis model: {e}")
    print("Sentiment analysis and related tasks will be skipped.")

# Load spaCy model - ensure it's downloaded first
download_spacy_model()
nlp = None
try:
    nlp = spacy.load('en_core_web_sm')
    print("spaCy 'en_core_web_sm' model loaded for analysis.")
except OSError:
    print("spaCy 'en_core_web_sm' model not found. Please run utils.download_spacy_model() or ensure it's installed.")
except Exception as e:
     print(f"An unexpected error occurred loading spaCy model for analysis: {e}")


def preprocess_text_spacy(text, spacy_model):
    """Basic text preprocessing using spaCy."""
    if spacy_model is None or not isinstance(text, str):
        return ""

    doc = spacy_model(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def identify_themes(processed_text, theme_keywords):
    """Identifies themes in a review based on keywords."""
    identified = set()
    processed_text_str = str(processed_text)
    for theme, keywords in theme_keywords.items():
        # Check for whole words or defined phrases
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', processed_text_str) for keyword in keywords):
             identified.add(theme)
    return list(identified)

def analyze_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs sentiment and thematic analysis on the cleaned review data.

    Args:
        df (pd.DataFrame): DataFrame containing cleaned review data.

    Returns:
        pd.DataFrame: DataFrame with analysis results (sentiment, processed text, themes).
    """
    print("\n--- Starting Sentiment and Thematic Analysis ---")

    if df.empty:
        print("Sentiment and Thematic Analysis skipped due to empty dataframe.")
        analysis_df = df.copy()
        # Ensure expected analysis columns exist even in an empty DF
        analysis_cols = ['sentiment_label', 'sentiment_score', 'processed_text', 'identified_themes']
        for col in analysis_cols:
             if col not in analysis_df.columns:
                  analysis_df[col] = pd.Series(dtype='object')
        return analysis_df


    analysis_df = df.copy()
    print(f"Starting analysis on {len(analysis_df)} reviews.")

    # --- Sentiment Analysis ---
    print("\nPerforming sentiment analysis...")
    if SENTIMENT_MODEL_READY:
        # Ensure review_text is string type and handle NaNs before sentiment analysis
        analysis_df['review_text'] = analysis_df['review_text'].astype(str).fillna('')
        review_texts = analysis_df['review_text'].tolist()

        try:
            batch_size = 64 # Adjust based on your system
            sentiment_results = []
            print(f"Analyzing sentiment for {len(review_texts)} reviews in batches of {batch_size}...")
            for i in tqdm(range(0, len(review_texts), batch_size), desc="Sentiment Analysis Batches"):
                batch = review_texts[i:i+batch_size]
                # Add truncation=True and padding=True to handle varying text lengths
                results = sentiment_pipeline(batch, truncation=True, padding=True)
                sentiment_results.extend(results)

            print("Sentiment analysis completed.")

            # Extract labels and scores
            analysis_df['sentiment_label'] = [result['label'] for result in sentiment_results]
            # Adjust score based on label for a unified score (0 to 1)
            analysis_df['sentiment_score'] = [
                result['score'] if result['label'] == 'POSITIVE' else (1 - result['score'])
                for result in sentiment_results
            ]

            print("Sentiment labels and scores added.")

        except Exception as e:
            print(f"An error occurred during sentiment analysis: {e}")
            # Fallback: add sentiment columns with default values if analysis failed
            if 'sentiment_label' not in analysis_df.columns: analysis_df['sentiment_label'] = None
            if 'sentiment_score' not in analysis_df.columns: analysis_df['sentiment_score'] = None
            # Keep SENTIMENT_MODEL_READY as False for plotting later
            SENTIMENT_MODEL_READY = False


    # --- Text Preprocessing for Thematic Analysis ---
    print("\nPreprocessing text for thematic analysis...")
    if nlp is not None: # Ensure spaCy model loaded
        # Ensure review_text is string type and handle NaNs
        analysis_df['review_text'] = analysis_df['review_text'].astype(str).fillna('')

        # Apply preprocessing (using tqdm for progress bar)
        tqdm.pandas(desc="Applying spaCy Preprocessing")
        analysis_df['processed_text'] = analysis_df['review_text'].progress_apply(
            lambda x: preprocess_text_spacy(x, nlp)
        )

        print("Text preprocessing complete.")
    else:
         print("Skipping text preprocessing as spaCy model failed to load.")
         # Add processed_text column with original text if preprocessing fails
         analysis_df['processed_text'] = analysis_df['review_text'].astype(str).fillna('')


    # --- Thematic Grouping (Using THEME_KEYWORDS from utils) ---
    print("\nIdentifying themes based on keywords...")

    # Apply theme identification (using tqdm with pandas apply)
    tqdm.pandas(desc="Identifying Themes")
    analysis_df['identified_themes'] = analysis_df['processed_text'].progress_apply(
        lambda x: identify_themes(x, THEME_KEYWORDS)
    )

    print("Theme identification complete.")
    print(f"\nSample reviews with identified themes:")
    # Only display if there are reviews
    if not analysis_df.empty:
        display(analysis_df[['review_text', 'identified_themes']].sample(min(5, len(analysis_df)))) # Show max 5 samples


    # --- Prepare data for Analysis Scenarios ---
    # This DataFrame `analysis_df` now contains review_id, original text, rating, date,
    # bank, source, date_normalized, app_id, sentiment_label, sentiment_score,
    # processed_text, and identified_themes. This is the input for scenario analysis.


    print("\n--- Sentiment and Thematic Analysis Complete ---")

    return analysis_df # Return the DataFrame with analysis results

def generate_scenario_1_plots(df: pd.DataFrame, banks: list):
    """
    Generates plots for Scenario 1: Retaining Users (Focus on Ratings and 'slow loading' issues).
    """
    print("\n--- Generating Scenario 1 Plots (Ratings and Performance Issues) ---")
    if df.empty:
        print("No data to generate Scenario 1 plots.")
        return

    # Filter for relevant theme (Performance Issues)
    performance_issue_theme = 'App Performance (Slow/Crash)'
    df_performance_issues = df[df['identified_themes'].apply(lambda themes: performance_issue_theme in themes)].copy()

    if df_performance_issues.empty:
         print(f"No reviews identified with the theme '{performance_issue_theme}'. Skipping related plots.")


    # Plot 1: Distribution of Ratings per Bank
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='rating', hue='bank_name', palette='viridis')
    plt.title('Distribution of Ratings per Bank')
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.legend(title='Bank')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'scenario1_rating_distribution_{TODAY_DATE_STR}.png'))
    plt.close()
    print("Plot 1: Rating distribution saved.")


    # Plot 2: Average Rating per Bank
    plt.figure(figsize=(8, 5))
    # Ensure rating is numeric before calculating mean
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    avg_ratings = df.groupby('bank_name')['rating'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_ratings.index, y=avg_ratings.values, palette='viridis')
    plt.title('Average Rating per Bank')
    plt.xlabel('Bank')
    plt.ylabel('Average Rating')
    plt.ylim(0, 5) # Set y-axis limit for ratings
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'scenario1_average_rating_{TODAY_DATE_STR}.png'))
    plt.close()
    print("Plot 2: Average rating per bank saved.")


    # Plot 3: Count of Reviews with 'App Performance (Slow/Crash)' Theme per Bank
    if not df_performance_issues.empty:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_performance_issues, x='bank_name', palette='viridis')
        plt.title(f"Count of Reviews with '{performance_issue_theme}' Theme per Bank")
        plt.xlabel('Bank')
        plt.ylabel('Number of Reviews')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'scenario1_performance_theme_count_{TODAY_DATE_STR}.png'))
        plt.close()
        print(f"Plot 3: '{performance_issue_theme}' theme count saved.")

    # Plot 4: Average Sentiment Score for Reviews with 'App Performance (Slow/Crash)' Theme per Bank
    if not df_performance_issues.empty and 'sentiment_score' in df_performance_issues.columns and SENTIMENT_MODEL_READY:
        plt.figure(figsize=(10, 6))
        # Ensure sentiment_score is numeric
        df_performance_issues['sentiment_score'] = pd.to_numeric(df_performance_issues['sentiment_score'], errors='coerce')
        avg_sentiment_performance = df_performance_issues.groupby('bank_name')['sentiment_score'].mean().sort_values() # Sort ascending for sentiment score (lower means more negative)
        sns.barplot(x=avg_sentiment_performance.index, y=avg_sentiment_performance.values, palette='viridis')
        plt.title(f"Average Sentiment Score for '{performance_issue_theme}' Reviews per Bank")
        plt.xlabel('Bank')
        plt.ylabel('Average Sentiment Score (0=Negative, 1=Positive)')
        plt.ylim(0, 1) # Sentiment score range
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'scenario1_performance_theme_sentiment_{TODAY_DATE_STR}.png'))
        plt.close()
        print(f"Plot 4: '{performance_issue_theme}' theme sentiment saved.")
    elif not SENTIMENT_MODEL_READY:
        print("Skipping sentiment plots for Scenario 1 as the sentiment model was not ready.")


    print("--- Scenario 1 Plots Generated ---")


def generate_scenario_2_plots(df: pd.DataFrame):
    """
    Generates plots for Scenario 2: Enhancing Features (Focus on Feature-related themes and keywords).
    """
    print("\n--- Generating Scenario 2 Plots (Feature Analysis) ---")
    if df.empty:
        print("No data to generate Scenario 2 plots.")
        return

    feature_suggestion_theme = 'Missing Features/Suggestions'
    df_features = df[df['identified_themes'].apply(lambda themes: feature_suggestion_theme in themes)].copy()

    if df_features.empty:
        print(f"No reviews identified with the theme '{feature_suggestion_theme}'. Skipping related plots.")
        return # Exit if no feature reviews


    # Plot 1: Count of Reviews with 'Missing Features/Suggestions' Theme per Bank
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_features, x='bank_name', palette='viridis')
    plt.title(f"Count of Reviews Mentioning '{feature_suggestion_theme}' per Bank")
    plt.xlabel('Bank')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'scenario2_feature_theme_count_{TODAY_DATE_STR}.png'))
    plt.close()
    print(f"Plot 1: '{feature_suggestion_theme}' theme count saved.")


    # Plot 2: Sentiment Distribution for 'Missing Features/Suggestions' Reviews
    if 'sentiment_label' in df_features.columns and SENTIMENT_MODEL_READY:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_features, x='sentiment_label', order=['NEGATIVE', 'POSITIVE'], palette='viridis')
        plt.title(f"Sentiment Distribution for '{feature_suggestion_theme}' Reviews")
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'scenario2_feature_theme_sentiment_dist_{TODAY_DATE_STR}.png'))
        plt.close()
        print(f"Plot 2: '{feature_suggestion_theme}' sentiment distribution saved.")
    elif not SENTIMENT_MODEL_READY:
        print("Skipping sentiment plots for Scenario 2 as the sentiment model was not ready.")


    # Plot 3: Top Keywords in 'Missing Features/Suggestions' Reviews (Word Cloud)
    if not df_features['processed_text'].empty and df_features['processed_text'].str.strip().any():
        text_for_wordcloud = " ".join(df_features['processed_text'].dropna().tolist())
        # Exclude theme keywords themselves from the word cloud if desired
        stopwords_wc = set(STOPWORDS).union(set(keyword for keywords in THEME_KEYWORDS.values() for keyword in keywords))

        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_wc).generate(text_for_wordcloud)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Top Keywords in '{feature_suggestion_theme}' Reviews")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'scenario2_feature_theme_wordcloud_{TODAY_DATE_STR}.png'))
        plt.close()
        print(f"Plot 3: '{feature_suggestion_theme}' word cloud saved.")
    else:
        print("No processed text available for Word Cloud in Scenario 2.")


    print("--- Scenario 2 Plots Generated ---")


def generate_scenario_3_plots(df: pd.DataFrame):
    """
    Generates plots for Scenario 3: Managing Complaints (Focus on Complaint-related themes).
    """
    print("\n--- Generating Scenario 3 Plots (Complaint Analysis) ---")
    if df.empty:
        print("No data to generate Scenario 3 plots.")
        return

    # Identify complaint-related themes (example themes, adjust as needed)
    complaint_themes = ['Login/Account Issues', 'Transaction Problems', 'App Performance (Slow/Crash)', 'Customer Support', 'Network/Connectivity']
    df_complaints = df[
        df['identified_themes'].apply(lambda themes: any(theme in themes for theme in complaint_themes))
    ].copy()

    if df_complaints.empty:
        print("No reviews identified with common complaint themes. Skipping related plots.")
        return

    # Plot 1: Overall Complaint Theme Distribution
    all_complaint_themes = [theme for themes_list in df_complaints['identified_themes'].dropna() for theme in themes_list if theme in complaint_themes]
    if all_complaint_themes:
        theme_counts = Counter(all_complaint_themes)
        themes_df = pd.DataFrame.from_dict(theme_counts, orient='index', columns=['count']).sort_values(by='count', ascending=False)

        plt.figure(figsize=(12, 7))
        sns.barplot(x=themes_df.index, y=themes_df['count'], palette='viridis')
        plt.title('Overall Distribution of Complaint Themes')
        plt.xlabel('Complaint Theme')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'scenario3_overall_complaint_theme_distribution_{TODAY_DATE_STR}.png'))
        plt.close()
        print("Plot 1: Overall complaint theme distribution saved.")
    else:
        print("No complaint themes found for plotting overall distribution.")


    # Plot 2: Complaint Theme Distribution per Bank
    if not df_complaints.empty:
        bank_theme_counts = df_complaints.groupby('bank_name')['identified_themes'].apply(
            lambda x: [theme for themes_list in x.dropna() for theme in themes_list if theme in complaint_themes]
        )
        bank_theme_flat = [(bank, theme) for bank, themes_list in bank_theme_counts.items() for theme in themes_list]
        if bank_theme_flat:
            bank_theme_df = pd.DataFrame(bank_theme_flat, columns=['bank', 'theme'])
            plt.figure(figsize=(14, 8))
            sns.countplot(data=bank_theme_df, x='theme', hue='bank', palette='viridis')
            plt.title('Distribution of Complaint Themes per Bank')
            plt.xlabel('Complaint Theme')
            plt.ylabel('Number of Reviews')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Bank')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'scenario3_complaint_theme_distribution_per_bank_{TODAY_DATE_STR}.png'))
            plt.close()
            print("Plot 2: Complaint theme distribution per bank saved.")
        else:
            print("No complaint themes found per bank for plotting.")


    # Plot 3: Sentiment Distribution for Complaint Reviews
    if 'sentiment_label' in df_complaints.columns and SENTIMENT_MODEL_READY:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_complaints, x='sentiment_label', order=['NEGATIVE', 'POSITIVE'], palette='viridis')
        plt.title("Sentiment Distribution for Reviews with Complaint Themes")
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'scenario3_complaint_sentiment_dist_{TODAY_DATE_STR}.png'))
        plt.close()
        print("Plot 3: Complaint review sentiment distribution saved.")
    elif not SENTIMENT_MODEL_READY:
         print("Skipping sentiment plots for Scenario 3 as the sentiment model was not ready.")

    # Plot 4: Top Keywords in Reviews with Complaint Themes (Word Cloud)
    if not df_complaints['processed_text'].empty and df_complaints['processed_text'].str.strip().any():
        text_for_wordcloud = " ".join(df_complaints['processed_text'].dropna().tolist())
        # Exclude complaint theme keywords themselves
        stopwords_wc = set(STOPWORDS).union(set(keyword for themes in [THEME_KEYWORDS[theme] for theme in complaint_themes if theme in THEME_KEYWORDS] for keyword in themes))

        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_wc).generate(text_for_wordcloud)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Top Keywords in Reviews with Complaint Themes")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'scenario3_complaint_wordcloud_{TODAY_DATE_STR}.png'))
        plt.close()
        print("Plot 4: Complaint review word cloud saved.")
    else:
        print("No processed text available for Word Cloud in Scenario 3.")


    print("--- Scenario 3 Plots Generated ---")

# You could add a general EDA function here if absolutely necessary,
# but focus should be on the scenario-specific plots as per the task.
# Example:
def generate_general_eda_plots(df: pd.DataFrame):
    """Generates some basic general EDA plots."""
    print("\n--- Generating General EDA Plots ---")
    if df.empty:
        print("No data to generate general EDA plots.")
        return

    # Example: Overall Sentiment Distribution
    if 'sentiment_label' in df.columns and SENTIMENT_MODEL_READY:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='sentiment_label', order=['NEGATIVE', 'POSITIVE'], palette='viridis')
        plt.title('Overall Sentiment Distribution of Reviews')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'general_overall_sentiment_dist_{TODAY_DATE_STR}.png'))
        plt.close()
        print("General EDA Plot: Overall sentiment distribution saved.")
    elif not SENTIMENT_MODEL_READY:
         print("Skipping general sentiment plot as the sentiment model was not ready.")

    # Example: Word Cloud of most frequent words (excluding stopwords and theme keywords)
    if not df['processed_text'].empty and df['processed_text'].str.strip().any():
         text_for_wordcloud = " ".join(df['processed_text'].dropna().tolist())
         # Combine NLTK stopwords, WordCloud default stopwords, and theme keywords
         all_stopwords = set(STOPWORDS).union(set(keyword for keywords in THEME_KEYWORDS.values() for keyword in keywords))
         try:
              # Need NLTK stopwords if not already included in the combined set
              from nltk.corpus import stopwords
              nltk_stopwords = set(stopwords.words('english'))
              all_stopwords = all_stopwords.union(nltk_stopwords)
         except LookupError:
              print("NLTK stopwords not found. Skipping adding them to word cloud stopwords.")
         except Exception as e:
              print(f"Error loading NLTK stopwords: {e}")

         wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=all_stopwords).generate(text_for_wordcloud)
         plt.figure(figsize=(10, 5))
         plt.imshow(wordcloud, interpolation='bilinear')
         plt.axis("off")
         plt.title("Overall Top Keywords in Reviews")
         plt.tight_layout()
         plt.savefig(os.path.join(FIGURES_DIR, f'general_overall_wordcloud_{TODAY_DATE_STR}.png'))
         plt.close()
         print("General EDA Plot: Overall word cloud saved.")
    else:
         print("No processed text available for general Word Cloud.")

    print("--- General EDA Plots Generated ---")


if __name__ == '__main__':
    # Example usage if you run analysis.py directly
    # You would typically run this from main.py after scraping and preprocessing
    print("Running analysis.py directly is for testing purposes.")
    print("Please ensure you have a 'data/cleaned' directory with a CSV file like 'all_reviews_cleaned_YYYYMMDD.csv'")

    # Create dummy data for testing
    dummy_data = {
        'review_id': [0, 1, 2, 3, 4, 5, 6],
        'review_text': [
            'The app is great and fast.',
            'Slow loading times and crashes.',
            'Login issues all the time.',
            'I need fingerprint login feature.',
            'Customer support was helpful.',
            'Transaction failed.',
            'Nice UI design.'
        ],
        'processed_text': [ # Add processed text for theme identification
            'app great fast',
            'slow load time crash',
            'login issue time',
            'need fingerprint login feature',
            'customer support helpful',
            'transaction fail',
            'nice ui design'
        ],
        'rating': [5, 1, 1, 4, 5, 2, 4],
        'review_date': pd.to_datetime(['2023-10-27', '2023-10-27', '2023-10-27', '2023-10-28', '2023-10-28', '2023-10-29', '2023-10-29']),
        'bank_name': ['Commercial_Bank_of_Ethiopia', 'Bank_of_Abyssinia', 'Dashen_Bank_Superapp', 'Commercial_Bank_of_Ethiopia', 'Bank_of_Abyssinia', 'Dashen_Bank_Superapp', 'Commercial_Bank_of_Ethiopia'],
        'source': ['Google Play'] * 7,
        'date_normalized': ['2023-10-27', '2023-10-27', '2023-10-27', '2023-10-28', '2023-10-28', '2023-10-29', '2023-10-29'],
        'app_id': ['com.cbe', 'com.boa', 'com.dashen', 'com.cbe', 'com.boa', 'com.dashen', 'com.cbe']
    }
    dummy_df = pd.DataFrame(dummy_data)

    # Ensure directories exist
    from utils import create_directories
    create_directories()

    # Run analysis on dummy data
    analysis_results_df = analyze_reviews(dummy_df)
    print("\nAnalysis results (dummy data):")
    print(analysis_results_df[['review_text', 'sentiment_label', 'sentiment_score', 'identified_themes']].head())

    # Generate plots for scenarios (using dummy data)
    generate_scenario_1_plots(analysis_results_df, list(analysis_results_df['bank_name'].unique()))
    generate_scenario_2_plots(analysis_results_df)
    generate_scenario_3_plots(analysis_results_df)
    generate_general_eda_plots(analysis_results_df) # Include general EDA for testing