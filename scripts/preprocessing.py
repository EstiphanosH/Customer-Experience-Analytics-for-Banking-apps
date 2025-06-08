import pandas as pd
import os
from tqdm import tqdm
import logging
import oracledb # Import the Oracle DB driver
import re # Import regex for future potential use, but using remove_non_printable now

from utils import CLEANED_DATA_DIR, APP_ID_TO_BANK_NAME, TODAY_DATE_STR, ORACLE_DB_USER, ORACLE_DB_PASSWORD, ORACLE_DB_DSN, ORACLE_TABLE_NAME, remove_non_printable # Import the new function and DB constants

class ReviewPreprocessor:
    def __init__(self, cleaned_data_dir=CLEANED_DATA_DIR, app_id_to_bank_name=APP_ID_TO_BANK_NAME,
                 db_user=ORACLE_DB_USER, db_password=ORACLE_DB_PASSWORD, db_dsn=ORACLE_DB_DSN, db_table=ORACLE_TABLE_NAME):
        """
        Initializes the ReviewPreprocessor.

        Args:
            cleaned_data_dir (str): Directory where cleaned data will be saved.
            app_id_to_bank_name (dict): Dictionary mapping app IDs to bank names.
            db_user (str): Oracle database username.
            db_password (str): Oracle database password.
            db_dsn (str): Oracle database connection string (DSN).
            db_table (str): Name of the Oracle table to save data to.
        """
        self.cleaned_data_dir = cleaned_data_dir
        self.app_id_to_bank_name = app_id_to_bank_name
        self.df: pd.DataFrame | None = None # Use type hint for clarity

        self.db_user = db_user
        self.db_password = db_password
        self.db_dsn = db_dsn
        self.db_table = db_table

        # Define the columns as they are expected *after* initial loading and renaming
        # Assuming raw CSV has 'review_text', 'rating', 'date', 'app_id', 'source'
        self.internal_cols = ['review_text', 'rating', 'review_date', 'app_id', 'source'] # Standardized internal names

        # Define the final columns to be saved (matching desired DB schema)
        # Added processed_text as it's useful for analysis and might be stored
        self.final_db_cols = ['review_id', 'review_text', 'processed_text', 'rating', 'review_date', 'bank_name', 'source', 'date_normalized', 'app_id']


        os.makedirs(self.cleaned_data_dir, exist_ok=True)
        # Configure logging for this module specifically if needed, or rely on basicConfig in utils/main
        # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("ReviewPreprocessor initialized.")


    def load_data(self, raw_filepath: str) -> bool:
        """
        Loads raw data from a CSV file into a DataFrame and applies initial renaming.

        Args:
            raw_filepath (str): Path to the raw CSV file.

        Returns:
            bool: True if data loaded successfully and required columns exist.
        """
        try:
            # Use 'utf-8' encoding, add error handling for decoding issues
            self.df = pd.read_csv(raw_filepath, encoding='utf-8', errors='replace')
            logging.info(f"Loaded data from {raw_filepath}. Initial shape: {self.df.shape}")

            # Ensure the columns expected from the raw CSV are present
            expected_raw_cols = ['review_text', 'rating', 'date', 'app_id', 'source']
            missing_raw_cols = [col for col in expected_raw_cols if col not in self.df.columns]
            if missing_raw_cols:
                logging.error(f"Missing expected raw columns in {raw_filepath}: {missing_raw_cols}. Cannot proceed.")
                self.df = pd.DataFrame()
                return False

            # Rename to internal standard names
            rename_map = {
                'review_text': 'review_text',
                'rating': 'rating',
                'date': 'review_date',
                'app_id': 'app_id',
                'source': 'source'
            }
            # Apply renaming only if the source column exists
            cols_to_rename = {k: v for k, v in rename_map.items() if k in self.df.columns}
            self.df.rename(columns=cols_to_rename, inplace=True)

            # Check if the internal standard columns are now present
            if not all(col in self.df.columns for col in self.internal_cols):
                 logging.error(f"Internal standard columns {self.internal_cols} not present after renaming. Cannot proceed.")
                 self.df = pd.DataFrame()
                 return False

            return True

        except FileNotFoundError:
            logging.error(f"Raw data file not found at {raw_filepath}")
            self.df = pd.DataFrame()
            return False
        except Exception as e:
            logging.error(f"Error loading data from {raw_filepath}: {e}")
            self.df = pd.DataFrame()
            return False

    def clean_review_text(self):
        """Applies text cleaning steps: remove non-printable chars."""
        if self.df is None or self.df.empty or 'review_text' not in self.df.columns:
            logging.warning("No data loaded, DataFrame is empty, or 'review_text' column missing. Skipping text cleaning.")
            return

        logging.info("Cleaning 'review_text' (removing non-printable characters)...")
        # Ensure review_text is string type before applying the cleaning function
        self.df['review_text'] = self.df['review_text'].astype(str).apply(remove_non_printable)
        logging.info("'review_text' cleaning complete.")


    def normalize_dates(self):
        """Converts date column to datetime objects and adds a normalized date column."""
        if self.df is None or self.df.empty:
            logging.warning("No data loaded or DataFrame is empty. Skipping date normalization.")
            return
        # Use the *internal* standard date column name
        if 'review_date' not in self.df.columns:
             logging.error("'review_date' column not found. Skipping date normalization.")
             return

        original_dtype = self.df['review_date'].dtype
        self.df['review_date'] = pd.to_datetime(self.df['review_date'], errors='coerce')

        if not pd.api.types.is_datetime64_any_dtype(self.df['review_date']):
             logging.warning(f"Could not convert 'review_date' column to datetime from {original_dtype}. It remains type {self.df['review_date'].dtype}. Normalization skipped.")
             # Add the normalized column anyway, but it might contain NaT or errors
             self.df['date_normalized'] = None # Or a placeholder
        else:
            self.df['date_normalized'] = self.df['review_date'].dt.strftime('%Y-%m-%d')
            logging.info("'review_date' column converted to datetime and 'date_normalized' column added.")

    def handle_missing_values(self):
        """Drops rows with missing essential data ('review_text', 'rating', 'review_date')."""
        if self.df is None or self.df.empty:
            logging.warning("No data loaded or DataFrame is empty. Skipping missing value handling.")
            return

        check_cols = ['review_text', 'rating', 'review_date']
        # Ensure these columns exist before trying to drop NaNs
        if not all(col in self.df.columns for col in check_cols):
             logging.error(f"Essential columns {check_cols} not found. Cannot handle missing values.")
             # Optionally, reset df to empty if essential columns are gone
             # self.df = pd.DataFrame()
             return

        initial_rows = len(self.df)
        # Drop rows with missing 'review_text' or 'rating'
        self.df.dropna(subset=['review_text', 'rating'], inplace=True)
        logging.info(f"Dropped {initial_rows - len(self.df)} rows with missing 'review_text' or 'rating'. New shape: {self.df.shape}")

        initial_rows = len(self.df)
        # Drop rows where date conversion resulted in NaT (handled by normalize_dates)
        self.df.dropna(subset=['review_date'], inplace=True)
        logging.info(f"Dropped {initial_rows - len(self.df)} rows with invalid dates. New shape: {self.df.shape}")

    def remove_duplicates(self):
        """Removes duplicate rows based on essential columns."""
        if self.df is None or self.df.empty:
            logging.warning("No data loaded or DataFrame is empty. Skipping duplicate removal.")
            return

        subset_cols = ['app_id', 'review_text', 'rating', 'review_date']
        if not all(col in self.df.columns for col in subset_cols):
             logging.error(f"Essential columns {subset_cols} not found. Cannot remove duplicates.")
             # Optionally, reset df to empty
             # self.df = pd.DataFrame()
             return

        initial_rows = len(self.df)
        self.df.drop_duplicates(subset=subset_cols, inplace=True)
        logging.info(f"Dropped {initial_rows - len(self.df)} duplicate rows based on {subset_cols}. New shape: {self.df.shape}")

    def add_bank_name(self):
        """Adds the 'bank_name' column by mapping app IDs."""
        if self.df is None or self.df.empty:
            logging.warning("No data loaded or DataFrame is empty. Skipping adding bank name.")
            return
        if 'app_id' not in self.df.columns:
             logging.error("'app_id' column not found. Cannot add bank name.")
             # Add the column with a default value to maintain structure
             self.df['bank_name'] = 'Unknown_Bank'
             return

        self.df['bank_name'] = self.df['app_id'].map(self.app_id_to_bank_name).fillna('Unknown_Bank')
        logging.info("'bank_name' column added.")

    def finalize_columns(self):
        """Ensures final output columns exist and are in the correct order, adds review_id."""
        if self.df is None or self.df.empty:
            logging.warning("No data loaded or DataFrame is empty. Skipping column finalization.")
            return

        # Add review_id as a sequential ID for the *current* state of the DataFrame
        if 'review_id' not in self.df.columns:
             self.df['review_id'] = range(len(self.df))
             logging.info("'review_id' column added.")

        # Ensure 'processed_text' is added if not already there (it will be added in analysis, but ensure it exists)
        if 'processed_text' not in self.df.columns:
            self.df['processed_text'] = '' # Add with empty string default

        # Ensure all final database columns exist before selecting
        for col in self.final_db_cols:
             if col not in self.df.columns:
                 default_value = None
                 if col in ['review_text', 'processed_text', 'bank_name', 'source', 'app_id', 'date_normalized']:
                     default_value = '' # String default
                 elif col in ['rating', 'review_id']:
                      default_value = -1 # Numeric placeholder

                 logging.warning(f"Final DB column '{col}' not found. Adding with default value: {default_value}")
                 self.df[col] = default_value

        # Select and reorder the final database columns
        # Ensure columns to select are actually in the DataFrame after adding defaults
        cols_to_select = [col for col in self.final_db_cols if col in self.df.columns]
        self.df = self.df[cols_to_select].copy()

        logging.info("Columns finalized for database saving.")


    def save_cleaned_data_to_csv(self, df_to_save: pd.DataFrame, filename_suffix: str = TODAY_DATE_STR) -> str | None:
        """
        Saves a given DataFrame to a CSV file.

        Args:
            df_to_save (pd.DataFrame): The DataFrame to save.
            filename_suffix (str): Suffix for the filename (default is today's date).

        Returns:
            str | None: The path of the saved file if successful, None otherwise.
        """
        if df_to_save is None or df_to_save.empty:
            logging.warning("No cleaned data to save to CSV.")
            return None

        # Determine bank name for filename - use the 'bank_name' column from the DF if available
        bank_name_for_filename = 'Unknown_Bank'
        if 'bank_name' in df_to_save.columns and not df_to_save.empty:
             bank_name_for_filename = df_to_save['bank_name'].mode()[0] if not df_to_save['bank_name'].mode().empty else 'Unknown_Bank'

        bank_name_for_filename = bank_name_for_filename.replace(' ', '_')

        cleaned_filename = f'{bank_name_for_filename}_cleaned_{filename_suffix}.csv'
        cleaned_filepath = os.path.join(self.cleaned_data_dir, cleaned_filename)

        try:
            df_for_csv = df_to_save.copy()
            # Ensure the date column is in a string format for CSV if it's datetime objects
            if 'review_date' in df_for_csv.columns:
                 if pd.api.types.is_datetime64_any_dtype(df_for_csv['review_date']):
                    df_for_csv['review_date'] = df_for_csv['review_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                 else:
                    df_for_csv['review_date'] = df_for_csv['review_date'].astype(str)

            df_for_csv.to_csv(cleaned_filepath, index=False)
            logging.info(f"Successfully saved cleaned data to CSV: {cleaned_filepath}")
            return cleaned_filepath
        except Exception as e:
            logging.error(f"Error saving cleaned data to CSV {cleaned_filepath}: {e}")
            return None

    def save_cleaned_data_to_oracle(self, df_to_save: pd.DataFrame):
        """
        Saves the cleaned DataFrame to an Oracle database table.
        Assumes the table structure matches the final_db_cols.
        """
        if df_to_save is None or df_to_save.empty:
            logging.warning("No cleaned data to save to Oracle.")
            return

        if self.db_user == 'your_db_user' or self.db_password == 'your_db_password' or self.db_dsn == 'your_host:your_port/your_service_name':
             logging.warning("Oracle database credentials are not set. Skipping database save.")
             logging.warning("Please update ORACLE_DB_USER, ORACLE_DB_PASSWORD, and ORACLE_DB_DSN in utils.py or use environment variables.")
             return


        try:
            logging.info(f"Connecting to Oracle database: {self.db_dsn}...")
            connection = oracledb.connect(user=self.db_user, password=self.db_password, dsn=self.db_dsn)
            cursor = connection.cursor()
            logging.info("Oracle database connection successful.")

            data_to_insert = []
            for index, row in df_to_save.iterrows():
                review_date_py = row['review_date'].to_pydatetime() if pd.notnull(row['review_date']) else None
                review_text_str = str(row.get('review_text', '')) if pd.notnull(row.get('review_text')) else ''
                processed_text_str = str(row.get('processed_text', '')) if pd.notnull(row.get('processed_text')) else ''
                bank_name_str = str(row.get('bank_name', 'Unknown')) if pd.notnull(row.get('bank_name')) else 'Unknown'
                source_str = str(row.get('source', 'Unknown')) if pd.notnull(row.get('source')) else 'Unknown'
                date_normalized_str = str(row.get('date_normalized', '')) if pd.notnull(row.get('date_normalized')) else ''
                app_id_str = str(row.get('app_id', 'Unknown')) if pd.notnull(row.get('app_id')) else 'Unknown'

                rating_int = int(row['rating']) if pd.notnull(row['rating']) else -1
                review_id_int = int(row['review_id']) if pd.notnull(row['review_id']) else -1


                data_to_insert.append((
                    review_id_int,
                    review_text_str,
                    processed_text_str,
                    rating_int,
                    review_date_py,
                    bank_name_str,
                    source_str,
                    date_normalized_str,
                    app_id_str
                ))

            cols_for_insert = self.final_db_cols

            insert_sql = f"""
            INSERT INTO {self.db_table} ({', '.join(cols_for_insert)})
            VALUES ({', '.join([':' + str(i+1) for i in range(len(cols_for_insert))])})
            """
            logging.info(f"Insert SQL: {insert_sql}")

            batch_size = 1000
            logging.info(f"Inserting {len(data_to_insert)} rows into {self.db_table} in batches of {batch_size}...")
            cursor.executemany(insert_sql, data_to_insert, batcherrors=True)
            connection.commit()

            logging.info(f"Successfully inserted {len(data_to_insert)} rows into {self.db_table}.")

        except oracledb.Error as e:
            logging.error(f"Oracle database error during insertion: {e}")
            if hasattr(e, 'message') and 'batcherrors' in e.message:
                logging.error("Batch errors occurred:")
                for error in e.batcherrors:
                    logging.error(f"  Row {error.offset}: {error.message}")
            if connection:
                 connection.rollback()
        except Exception as e:
            logging.error(f"An unexpected error occurred during Oracle insertion: {e}")
            if connection:
                 connection.rollback()
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
            logging.info("Oracle database connection closed.")


    def preprocess_file(self, raw_filepath: str) -> pd.DataFrame | None:
        """
        Runs the preprocessing pipeline for a single raw data file.

        Args:
            raw_filepath (str): Path to the raw CSV file.

        Returns:
            pd.DataFrame | None: The cleaned DataFrame for the file, or None if failed or empty.
        """
        logging.info(f"\n--- Starting Preprocessing for {os.path.basename(raw_filepath)} ---")

        if not self.load_data(raw_filepath):
             logging.error(f"Skipping preprocessing for {os.path.basename(raw_filepath)} due to loading errors or missing columns.")
             return None

        if self.df.empty:
             logging.warning(f"DataFrame is empty after loading {os.path.basename(raw_filepath)}. Skipping subsequent steps.")
             return None

        self.clean_review_text() # Apply text cleaning (removes non-printables)
        self.normalize_dates()
        self.handle_missing_values()

        if self.df.empty:
             logging.warning(f"DataFrame is empty after handling missing values/invalid dates for {os.path.basename(raw_filepath)}. Skipping subsequent steps.")
             return None

        self.remove_duplicates()
        self.add_bank_name()
        # Finalize columns here to ensure the DataFrame returned by preprocess_file
        # has the consistent structure expected for concatenation and DB save.
        self.finalize_columns()

        logging.info(f"--- Preprocessing Complete for {os.path.basename(raw_filepath)} ---")

        # Return the cleaned DataFrame
        return self.df

    def preprocess_batch(self, raw_filepaths: list) -> pd.DataFrame:
        """
        Runs the preprocessing pipeline for a list of raw data files and concatenates results.

        Args:
            raw_filepaths (list): A list of paths to the raw CSV files.

        Returns:
            pd.DataFrame: The concatenated cleaned DataFrame.
        """
        print("\n--- Starting Batch Preprocessing ---")
        all_cleaned_dfs = []
        processed_csv_files = [] # List to store paths of saved individual cleaned CSVs

        for raw_filepath in tqdm(raw_filepaths, desc="Preprocessing Files"):
            cleaned_df = self.preprocess_file(raw_filepath)
            if cleaned_df is not None and not cleaned_df.empty:
                all_cleaned_dfs.append(cleaned_df)
                # Save each cleaned file individually to CSV (as done before)
                saved_csv_path = self.save_cleaned_data_to_csv(cleaned_df, filename_suffix=os.path.basename(raw_filepath).replace('.csv', '').replace('_raw', '_cleaned'))
                if saved_csv_path:
                    processed_csv_files.append(saved_csv_path)


        if all_cleaned_dfs:
            all_reviews_df = pd.concat(all_cleaned_dfs, ignore_index=True)
            # Re-add/update review_id for the combined dataframe
            all_reviews_df['review_id'] = range(len(all_reviews_df))
            print(f"\nSuccessfully combined all cleaned dataframes. Total reviews: {len(all_reviews_df)}")

            # Ensure the combined DataFrame has the final expected columns before saving
            # Temporarily assign to self.df to use finalize_columns
            temp_df = all_reviews_df.copy()
            self.df = temp_df
            self.finalize_columns() # This operates on self.df (temp_df)
            all_reviews_df = self.df.copy() # Get the finalized DataFrame back
            self.df = None # Reset self.df

            # Save the combined cleaned dataframe to a single CSV
            combined_cleaned_filepath = os.path.join(self.cleaned_data_dir, f'all_reviews_cleaned_{TODAY_DATE_STR}.csv')
            try:
                df_to_save_combined = all_reviews_df.copy()
                if 'review_date' in df_to_save_combined.columns:
                     if pd.api.types.is_datetime64_any_dtype(df_to_save_combined['review_date']):
                         df_to_save_combined['review_date'] = df_to_save_combined['review_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                     else:
                         df_to_save_combined['review_date'] = df_to_save_combined['review_date'].astype(str)

                df_to_save_combined.to_csv(combined_cleaned_filepath, index=False)
                print(f"Combined cleaned data saved to CSV: {combined_cleaned_filepath}")
            except Exception as e:
                print(f"Error saving combined cleaned data to CSV: {e}")

        else:
            print("\nNo cleaned dataframes were generated. Initializing an empty all_reviews_df.")
            all_reviews_df = pd.DataFrame()
            expected_cols = self.final_db_cols # Use the final column names
            for col in expected_cols:
                 all_reviews_df[col] = pd.Series(dtype='object')


        print("\n--- Batch Preprocessing Complete ---")
        print(f"Cleaned CSV files generated (individual): {processed_csv_files}")
        print(f"Shape of combined DataFrame: {all_reviews_df.shape}")

        return all_reviews_df # Return the combined DataFrame

if __name__ == '__main__':
    # Example usage if you run preprocessing.py directly
    from utils import create_directories
    import csv
    create_directories()
    # Create some dummy raw files for testing
    dummy_raw_dir = os.path.join("data", "raw")
    os.makedirs(dummy_raw_dir, exist_ok=True)
    # Include some non-printable characters (e.g., a null byte \x00, an escape character \x1b)
    with open(os.path.join(dummy_raw_dir, "Bank_A_raw_20231027.csv"), "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['review_text', 'rating', 'date', 'app_id', 'source'])
        writer.writerow(['Great\x00 app!', 5, '2023-10-27 10:00:00', 'com.banka.app', 'Google Play'])
        writer.writerow(['Slow loading \x1b and crashing.', 2, '2023-10-27 10:05:00', 'com.banka.app', 'Google Play'])
    with open(os.path.join(dummy_raw_dir, "Bank_B_raw_20231027.csv"), "w", newline="", encoding='utf-8') as f:
         writer = csv.writer(f)
         writer.writerow(['review_text', 'rating', 'date', 'app_id', 'source'])
         writer.writerow(['Very slow transaction. \tOften fails.', 1, '2023-10-27 11:00:00', 'com.bankb.app', 'Google Play']) # \t is printable whitespace
         writer.writerow(['Login issue always.\r\nRequires reinstall.', 1, '2023-10-27 11:10:00', 'com.bankb.app', 'Google Play']) # \r\n are printable whitespace


    preprocessor = ReviewPreprocessor(
        app_id_to_bank_name={'com.banka.app': 'Bank_A', 'com.bankb.app': 'Bank_B'},
        db_user='your_test_user', # Use dummy credentials for test
        db_password='your_test_password',
        db_dsn='your_test_host:your_test_port/your_test_service',
        db_table='TEST_APP_REVIEWS'
    )
    raw_files = [os.path.join(dummy_raw_dir, "Bank_A_raw_20231027.csv"), os.path.join(dummy_raw_dir, "Bank_B_raw_20231027.csv")]
    combined_df = preprocessor.preprocess_batch(raw_files)
    print("\nCombined DataFrame after preprocessing (check 'review_text' for removed non-printable chars):")
    print(combined_df.head())

    # Example of saving to database (will likely fail without actual DB connection)
    # print("\nAttempting to save combined data to Oracle...")
    # preprocessor.save_cleaned_data_to_oracle(combined_df) # Pass the dataframe