import os
import re
import logging
import pandas as pd
import numpy as np
import oracledb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bank_reviews_processing.log'),
        logging.StreamHandler()
    ]
)

class BankReviewsProcessor:
    def __init__(
        self,
        raw_data_dir: str = 'data/raw',
        processed_data_dir: str = 'data/processed',
        db_config: Dict = None,
        max_text_length: int = 4000,
        batch_size: int = 1000
    ):
        """
        Initialize the Bank Reviews Processing System.
        
        Args:
            raw_data_dir: Directory containing raw review files
            processed_data_dir: Directory for processed outputs
            db_config: Dictionary with database configuration
            max_text_length: Maximum length for text fields in database
            batch_size: Number of records per database batch insert
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.db_config = db_config or {
            'user': 'system',
            'password': 'oracle',
            'dsn': 'localhost:1521/XE',
            'schema': 'bank_reviews'
        }
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Database table definitions
        self.db_schema = {
            'banks': {
                'columns': [
                    ('bank_id', 'NUMBER', 'PRIMARY KEY'),
                    ('bank_name', 'VARCHAR2(100)', 'NOT NULL'),
                    ('app_id', 'VARCHAR2(100)', 'UNIQUE'),
                    ('source', 'VARCHAR2(50)', 'NOT NULL'),
                    ('date_added', 'DATE', 'DEFAULT SYSDATE')
                ],
                'constraints': []
            },
            'reviews': {
                'columns': [
                    ('review_id', 'NUMBER', 'PRIMARY KEY'),
                    ('bank_id', 'NUMBER', 'REFERENCES banks(bank_id)'),
                    ('review_text', 'VARCHAR2(4000)', 'NOT NULL'),
                    ('processed_text', 'VARCHAR2(4000)'),
                    ('rating', 'NUMBER(2)', 'CHECK (rating BETWEEN 1 AND 5)'),
                    ('review_date', 'DATE', 'NOT NULL'),
                    ('date_normalized', 'VARCHAR2(10)'),
                    ('source', 'VARCHAR2(50)', 'NOT NULL'),
                    ('sentiment_score', 'NUMBER(5,3)'),
                    ('sentiment_polarity', 'VARCHAR2(10)'),
                    ('thematic_category', 'VARCHAR2(100)'),
                    ('date_processed', 'DATE', 'DEFAULT SYSDATE')
                ],
                'constraints': [
                    'CONSTRAINT fk_bank FOREIGN KEY (bank_id) REFERENCES banks(bank_id)'
                ]
            }
        }
        
        # Final output columns
        self.output_columns = [
            'review_id', 'bank_name', 'review_text', 'processed_text',
            'rating', 'review_date', 'date_normalized', 'source',
            'sentiment_score', 'sentiment_polarity', 'thematic_category'
        ]
        
        logging.info("BankReviewsProcessor initialized successfully")

    def _get_db_connection(self) -> oracledb.Connection:
        """Establish database connection with error handling."""
        try:
            return oracledb.connect(
                user=self.db_config['user'],
                password=self.db_config['password'],
                dsn=self.db_config['dsn']
            )
        except oracledb.Error as e:
            logging.error(f"Database connection failed: {e}")
            raise

    def setup_database(self) -> bool:
        """
        Create database schema and tables if they don't exist.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Create schema if it doesn't exist
                    cursor.execute(f"""
                    DECLARE
                        schema_exists NUMBER;
                    BEGIN
                        SELECT COUNT(*) INTO schema_exists 
                        FROM ALL_USERS 
                        WHERE USERNAME = UPPER('{self.db_config['schema']}');
                        
                        IF schema_exists = 0 THEN
                            EXECUTE IMMEDIATE 'CREATE USER {self.db_config['schema']} 
                                IDENTIFIED BY {self.db_config['password']} 
                                DEFAULT TABLESPACE users 
                                TEMPORARY TABLESPACE temp 
                                QUOTA UNLIMITED ON users';
                            EXECUTE IMMEDIATE 'GRANT CONNECT, RESOURCE TO {self.db_config['schema']}';
                            DBMS_OUTPUT.PUT_LINE('Schema created successfully');
                        ELSE
                            DBMS_OUTPUT.PUT_LINE('Schema already exists');
                        END IF;
                    END;
                    """)
                    
                    # Switch to the schema
                    cursor.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {self.db_config['schema']}")
                    
                    # Create tables
                    for table_name, table_def in self.db_schema.items():
                        # Check if table exists
                        table_exists = cursor.execute(f"""
                            SELECT COUNT(*) FROM user_tables 
                            WHERE table_name = UPPER('{table_name}')
                        """).fetchone()[0]
                        
                        if not table_exists:
                            # Build CREATE TABLE statement
                            columns = [f"{col[0]} {col[1]} {col[2]}" for col in table_def['columns']]
                            constraints = table_def.get('constraints', [])
                            create_stmt = f"""
                                CREATE TABLE {table_name} (
                                    {', '.join(columns + constraints)}
                                )
                            """
                            cursor.execute(create_stmt)
                            logging.info(f"Created table {table_name}")
                            
                            # Create sequence for primary key
                            cursor.execute(f"""
                                CREATE SEQUENCE {table_name}_seq
                                START WITH 1
                                INCREMENT BY 1
                                NOCACHE
                                NOCYCLE
                            """)
                            
                            # Create trigger for auto-increment
                            cursor.execute(f"""
                                CREATE OR REPLACE TRIGGER {table_name}_trg
                                BEFORE INSERT ON {table_name}
                                FOR EACH ROW
                                BEGIN
                                    IF :NEW.{table_def['columns'][0][0]} IS NULL THEN
                                        :NEW.{table_def['columns'][0][0]} := {table_name}_seq.NEXTVAL;
                                    END IF;
                                END;
                            """)
                            
                    conn.commit()
                    logging.info("Database setup completed successfully")
                    return True
        except Exception as e:
            logging.error(f"Database setup failed: {e}")
            return False

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with comprehensive processing."""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special chars except basic punctuation
        
        # Normalize unicode and case
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = text.lower()
        
        return text[:self.max_text_length]

    def _normalize_date(self, date_str: str) -> Tuple[Optional[datetime], Optional[str]]:
        """Normalize date string to datetime object and YYYY-MM-DD format."""
        if pd.isna(date_str) or not str(date_str).strip():
            return None, None
            
        try:
            # Try multiple date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%b %d, %Y', '%Y/%m/%d']:
                try:
                    dt = datetime.strptime(str(date_str).strip(), fmt)
                    return dt, dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
                    
            # Fallback to pandas parser
            dt = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(dt):
                return dt.to_pydatetime(), dt.strftime('%Y-%m-%d')
                
            return None, None
        except Exception:
            return None, None

    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Perform sentiment analysis on text."""
        if not text or not isinstance(text, str):
            return 0.0, 'neutral'
            
        analysis = TextBlob(text)
        score = analysis.sentiment.polarity
        
        if score > 0.2:
            return score, 'positive'
        elif score < -0.2:
            return score, 'negative'
        else:
            return score, 'neutral'

    def _identify_thematic_category(self, text: str) -> str:
        """Identify thematic category using LDA (simplified example)."""
        if not text or not isinstance(text, str):
            return 'uncategorized'
            
        # In production, you'd want a pre-trained model
        common_categories = {
            'customer service': ['service', 'support', 'help', 'assistance'],
            'mobile app': ['app', 'mobile', 'application', 'android', 'ios'],
            'fees': ['fee', 'charge', 'cost', 'price'],
            'transactions': ['transfer', 'payment', 'transaction', 'send money'],
            'account': ['account', 'balance', 'statement', 'deposit']
        }
        
        text_lower = text.lower()
        for category, keywords in common_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
                
        return 'uncategorized'

    def preprocess_reviews(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Preprocess raw reviews DataFrame with comprehensive cleaning.
        
        Args:
            df: Raw DataFrame containing reviews
            source: Source of the reviews (e.g., 'Google Play', 'App Store')
            
        Returns:
            Cleaned and processed DataFrame
        """
        if df.empty:
            logging.warning("Received empty DataFrame for preprocessing")
            return pd.DataFrame(columns=self.output_columns)
            
        try:
            # Initial validation
            required_cols = {'review_text', 'rating', 'date', 'app_id'}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Create working copy
            processed_df = df.copy()
            
            # 1. Handle missing data
            initial_count = len(processed_df)
            processed_df = processed_df.dropna(subset=['review_text', 'rating'])
            processed_df = processed_df[processed_df['review_text'].astype(str).str.strip() != '']
            logging.info(f"Removed {initial_count - len(processed_df)} rows with missing data")
            
            # 2. Clean and normalize text
            processed_df['review_text'] = processed_df['review_text'].astype(str).apply(self._clean_text)
            processed_df['processed_text'] = processed_df['review_text']  # Will be further processed later
            
            # 3. Normalize dates
            date_results = processed_df['date'].apply(self._normalize_date)
            processed_df['review_date'] = [dr[0] for dr in date_results]
            processed_df['date_normalized'] = [dr[1] for dr in date_results]
            
            # Drop rows with invalid dates
            processed_df = processed_df[processed_df['review_date'].notna()]
            
            # 4. Handle ratings
            processed_df['rating'] = pd.to_numeric(processed_df['rating'], errors='coerce')
            processed_df = processed_df[processed_df['rating'].between(1, 5, inclusive='both')]
            
            # 5. Remove duplicates
            initial_count = len(processed_df)
            processed_df = processed_df.drop_duplicates(
                subset=['app_id', 'review_text', 'rating', 'date_normalized']
            )
            logging.info(f"Removed {initial_count - len(processed_df)} duplicate reviews")
            
            # 6. Add metadata
            processed_df['source'] = source
            processed_df['review_id'] = range(1, len(processed_df) + 1)
            
            # 7. Sentiment analysis
            sentiment_results = processed_df['processed_text'].apply(self._analyze_sentiment)
            processed_df['sentiment_score'] = [sr[0] for sr in sentiment_results]
            processed_df['sentiment_polarity'] = [sr[1] for sr in sentiment_results]
            
            # 8. Thematic categorization
            processed_df['thematic_category'] = processed_df['processed_text'].apply(
                self._identify_thematic_category
            )
            
            # Ensure final columns
            for col in self.output_columns:
                if col not in processed_df.columns:
                    processed_df[col] = None
            
            return processed_df[self.output_columns]
            
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            return pd.DataFrame(columns=self.output_columns)

    def perform_eda(self, df: pd.DataFrame, bank_name: str = "All Banks") -> Dict:
        """
        Perform comprehensive exploratory data analysis on reviews.
        
        Args:
            df: Processed DataFrame containing reviews
            bank_name: Name of bank for plot titles
            
        Returns:
            Dictionary containing EDA results and plots
        """
        eda_results = {}
        
        if df.empty:
            logging.warning("Cannot perform EDA on empty DataFrame")
            return eda_results
            
        try:
            # 1. Basic Statistics
            eda_results['basic_stats'] = {
                'total_reviews': len(df),
                'date_range': (
                    df['review_date'].min().strftime('%Y-%m-%d'),
                    df['review_date'].max().strftime('%Y-%m-%d')
                ) if pd.notna(df['review_date'].min()) else (None, None),
                'sources': df['source'].value_counts().to_dict(),
                'unique_banks': df['bank_name'].nunique()
            }
            
            # 2. Rating Distribution
            plt.figure(figsize=(10, 6))
            rating_plot = sns.countplot(x='rating', data=df, palette='viridis')
            plt.title(f'Rating Distribution - {bank_name}')
            plt.xlabel('Rating (1-5 Stars)')
            plt.ylabel('Count')
            rating_plot_path = os.path.join(self.processed_data_dir, f'rating_dist_{bank_name}.png')
            plt.savefig(rating_plot_path, bbox_inches='tight')
            plt.close()
            eda_results['rating_plot'] = rating_plot_path
            
            # 3. Sentiment Analysis
            plt.figure(figsize=(10, 6))
            sentiment_plot = sns.countplot(x='sentiment_polarity', data=df, 
                                          order=['positive', 'neutral', 'negative'],
                                          palette='coolwarm')
            plt.title(f'Sentiment Distribution - {bank_name}')
            plt.xlabel('Sentiment Polarity')
            plt.ylabel('Count')
            sentiment_plot_path = os.path.join(self.processed_data_dir, f'sentiment_dist_{bank_name}.png')
            plt.savefig(sentiment_plot_path, bbox_inches='tight')
            plt.close()
            eda_results['sentiment_plot'] = sentiment_plot_path
            
            # 4. Temporal Analysis
            if pd.notna(df['review_date'].min()):
                df['review_month'] = df['review_date'].dt.to_period('M')
                monthly_counts = df.groupby('review_month').size()
                
                plt.figure(figsize=(12, 6))
                monthly_plot = monthly_counts.plot(kind='line', marker='o')
                plt.title(f'Reviews Over Time - {bank_name}')
                plt.xlabel('Month')
                plt.ylabel('Number of Reviews')
                plt.xticks(rotation=45)
                monthly_plot_path = os.path.join(self.processed_data_dir, f'monthly_reviews_{bank_name}.png')
                plt.savefig(monthly_plot_path, bbox_inches='tight')
                plt.close()
                eda_results['monthly_plot'] = monthly_plot_path
            
            # 5. Thematic Categories
            plt.figure(figsize=(12, 6))
            category_plot = sns.countplot(
                y='thematic_category', 
                data=df, 
                palette='plasma',
                order=df['thematic_category'].value_counts().index
            )
            plt.title(f'Thematic Categories - {bank_name}')
            plt.xlabel('Count')
            plt.ylabel('Category')
            category_plot_path = os.path.join(self.processed_data_dir, f'categories_{bank_name}.png')
            plt.savefig(category_plot_path, bbox_inches='tight')
            plt.close()
            eda_results['category_plot'] = category_plot_path
            
            # 6. Word Cloud
            text = ' '.join(df['processed_text'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Word Cloud - {bank_name}')
            plt.axis('off')
            wordcloud_path = os.path.join(self.processed_data_dir, f'wordcloud_{bank_name}.png')
            plt.savefig(wordcloud_path, bbox_inches='tight')
            plt.close()
            eda_results['wordcloud'] = wordcloud_path
            
            # 7. Text Length Analysis
            df['text_length'] = df['processed_text'].str.len()
            
            plt.figure(figsize=(10, 6))
            length_plot = sns.histplot(df['text_length'], bins=30, kde=True)
            plt.title(f'Review Length Distribution - {bank_name}')
            plt.xlabel('Text Length (characters)')
            plt.ylabel('Count')
            length_plot_path = os.path.join(self.processed_data_dir, f'text_length_{bank_name}.png')
            plt.savefig(length_plot_path, bbox_inches='tight')
            plt.close()
            eda_results['length_plot'] = length_plot_path
            
            # 8. Correlation Analysis
            if 'rating' in df.columns and 'sentiment_score' in df.columns:
                plt.figure(figsize=(8, 6))
                corr_plot = sns.scatterplot(x='rating', y='sentiment_score', data=df, alpha=0.6)
                plt.title(f'Rating vs. Sentiment - {bank_name}')
                corr_plot_path = os.path.join(self.processed_data_dir, f'correlation_{bank_name}.png')
                plt.savefig(corr_plot_path, bbox_inches='tight')
                plt.close()
                eda_results['correlation_plot'] = corr_plot_path
            
            logging.info(f"Completed EDA for {bank_name}")
            return eda_results
            
        except Exception as e:
            logging.error(f"Error during EDA: {e}")
            return {}

    def save_to_database(self, df: pd.DataFrame) -> bool:
        """
        Save processed reviews to Oracle database.
        
        Args:
            df: Processed DataFrame containing reviews
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        if df.empty:
            logging.warning("No data to save to database")
            return False
            
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Set schema
                    cursor.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {self.db_config['schema']}")
                    
                    # Process in batches
                    total_rows = len(df)
                    num_batches = (total_rows + self.batch_size - 1) // self.batch_size
                    successful_rows = 0
                    
                    for batch_num in tqdm(range(num_batches), desc="Saving to database"):
                        start_idx = batch_num * self.batch_size
                        end_idx = min((batch_num + 1) * self.batch_size, total_rows)
                        batch = df.iloc[start_idx:end_idx]
                        
                        # Prepare bank data
                        bank_data = batch[['bank_name', 'app_id', 'source']].drop_duplicates()
                        
                        # Insert banks and get IDs
                        bank_ids = {}
                        for _, row in bank_data.iterrows():
                            # Check if bank exists
                            cursor.execute("""
                                SELECT bank_id FROM banks 
                                WHERE app_id = :app_id AND source = :source
                            """, app_id=row['app_id'], source=row['source'])
                            
                            result = cursor.fetchone()
                            if result:
                                bank_ids[(row['app_id'], row['source'])] = result[0]
                            else:
                                # Insert new bank
                                cursor.execute("""
                                    INSERT INTO banks (bank_name, app_id, source)
                                    VALUES (:1, :2, :3)
                                    RETURNING bank_id INTO :4
                                """, [row['bank_name'], row['app_id'], row['source'], cursor.var(oracledb.NUMBER)])
                                
                                bank_id = cursor.fetchone()[0]
                                bank_ids[(row['app_id'], row['source'])] = bank_id
                        
                        # Prepare review data
                        review_data = []
                        for _, row in batch.iterrows():
                            bank_id = bank_ids[(row['app_id'], row['source'])]
                            
                            review_data.append([
                                bank_id,
                                row['review_text'][:self.max_text_length],
                                row['processed_text'][:self.max_text_length] if pd.notna(row['processed_text']) else None,
                                row['rating'],
                                row['review_date'],
                                row['date_normalized'],
                                row['source'],
                                row['sentiment_score'],
                                row['sentiment_polarity'],
                                row['thematic_category']
                            ])
                        
                        # Insert reviews
                        cursor.executemany("""
                            INSERT INTO reviews (
                                bank_id, review_text, processed_text, rating,
                                review_date, date_normalized, source,
                                sentiment_score, sentiment_polarity, thematic_category
                            ) VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10)
                        """, review_data)
                        
                        successful_rows += len(review_data)
                        conn.commit()
                    
                    logging.info(f"Successfully saved {successful_rows}/{total_rows} rows to database")
                    return successful_rows == total_rows
                    
        except Exception as e:
            logging.error(f"Database save failed: {e}")
            if conn:
                conn.rollback()
            return False

    def process_file(self, filepath: str, source: str) -> bool:
        """
        Process a single file from raw data to database.
        
        Args:
            filepath: Path to raw data file
            source: Source of the reviews
            
        Returns:
            bool: True if processing was successful
        """
        try:
            # Load raw data
            df = pd.read_csv(filepath)
            
            # Preprocess
            processed_df = self.preprocess_reviews(df, source)
            if processed_df.empty:
                logging.warning(f"No valid data after preprocessing for {filepath}")
                return False
            
            # Perform EDA
            bank_name = os.path.basename(filepath).split('_')[0]
            eda_results = self.perform_eda(processed_df, bank_name)
            
            # Save to CSV
            output_filename = f"{bank_name}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = os.path.join(self.processed_data_dir, output_filename)
            processed_df.to_csv(output_path, index=False)
            logging.info(f"Saved processed data to {output_path}")
            
            # Save to database
            if not self.save_to_database(processed_df):
                logging.error(f"Failed to save {bank_name} data to database")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error processing {filepath}: {e}")
            return False

    def process_all_files(self) -> bool:
        """Process all raw files in the input directory."""
        success = True
        
        # Get all raw files
        raw_files = [
            f for f in os.listdir(self.raw_data_dir) 
            if f.endswith('.csv') and not f.startswith('.')
        ]
        
        if not raw_files:
            logging.warning("No raw files found to process")
            return False
            
        # Process each file
        for filename in tqdm(raw_files, desc="Processing files"):
            filepath = os.path.join(self.raw_data_dir, filename)
            source = 'Google Play' if 'google' in filename.lower() else 'App Store'
            
            if not self.process_file(filepath, source):
                success = False
                
        # Perform combined EDA
        if success:
            self.perform_combined_eda()
            
        return success

    def perform_combined_eda(self):
        """Perform EDA on all combined processed data."""
        try:
            # Get all processed files
            processed_files = [
                os.path.join(self.processed_data_dir, f) 
                for f in os.listdir(self.processed_data_dir) 
                if f.endswith('.csv') and not f.startswith('.')
            ]
            
            if not processed_files:
                logging.warning("No processed files found for combined EDA")
                return
                
            # Combine all data
            combined_df = pd.concat(
                [pd.read_csv(f) for f in processed_files],
                ignore_index=True
            )
            
            if combined_df.empty:
                logging.warning("Combined DataFrame is empty")
                return
                
            # Perform EDA
            self.perform_eda(combined_df, "All Banks Combined")
            
            # Generate thematic analysis report
            self.generate_thematic_report(combined_df)
            
        except Exception as e:
            logging.error(f"Error during combined EDA: {e}")

    def generate_thematic_report(self, df: pd.DataFrame):
        """Generate detailed thematic analysis report."""
        try:
            if df.empty:
                return
                
            # Prepare data for LDA
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tfidf = vectorizer.fit_transform(df['processed_text'].dropna())
            
            # Apply LDA
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(tfidf)
            
            # Get top words for each topic
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
                topics.append({
                    'topic_id': topic_idx + 1,
                    'top_words': ', '.join(top_words)
                })
            
            # Save topics to file
            topics_df = pd.DataFrame(topics)
            report_path = os.path.join(self.processed_data_dir, 'thematic_analysis_report.csv')
            topics_df.to_csv(report_path, index=False)
            logging.info(f"Saved thematic analysis report to {report_path}")
            
        except Exception as e:
            logging.error(f"Error generating thematic report: {e}")

if __name__ == "__main__":
    # Example usage
    processor = BankReviewsProcessor(
        raw_data_dir='data/raw',
        processed_data_dir='data/processed',
        db_config={
            'user': 'bank_reviews',
            'password': 'password123',
            'dsn': 'localhost:1521/XE',
            'schema': 'bank_reviews'
        }
    )
    
    # Setup database (only need to run once)
    if not processor.setup_database():
        logging.error("Failed to setup database")
        exit(1)
    # Process all files
    if processor.process_all_files():
        logging.info("All files processed successfully")
    else:
        logging.error("Some files failed to process")