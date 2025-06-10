# scraper.py (continued)

import os
import csv
from tqdm import tqdm
from google_play_scraper import reviews_all, Sort
from datetime import datetime
import logging

# Set up logging for the scraper module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GooglePlayScraper:
    """
    A class to scrape Google Play store reviews for multiple apps.
    Designed to be usable as a module in a larger project.
    """
    def __init__(self, app_ids_map, raw_data_dir="data/raw"):
        """
        Initializes the scraper.

        Args:
            app_ids_map (dict): A dictionary mapping app IDs to desired names
                                (e.g., {'com.example.app': 'Example_App'}).
            raw_data_dir (str): Directory to save raw scraped data.
        """
        if not isinstance(app_ids_map, dict) or not app_ids_map:
            raise ValueError("app_ids_map must be a non-empty dictionary.")
        if not isinstance(raw_data_dir, str) or not raw_data_dir:
            raise ValueError("raw_data_dir must be a non-empty string.")

        self.app_ids_map = app_ids_map
        self.raw_data_dir = raw_data_dir
        self.today_date_str = datetime.now().strftime('%Y%m%d')

        # Create raw data directory if it doesn't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        logging.info(f"Initialized GooglePlayScraper. Saving raw data to: {self.raw_data_dir}")

    def scrape_app_reviews(self, app_id, lang='en', country='us', sort=Sort.NEWEST, sleep_milliseconds=100):
        """
        Scrapes reviews for a single app ID.

        Args:
            app_id (str): The Google Play app ID.
            lang (str): Language code for reviews.
            country (str): Country code for reviews.
            sort (Sort): Sorting order for reviews (e.g., Sort.NEWEST, Sort.RATING).
            sleep_milliseconds (int): Time to sleep between requests to avoid rate limits.

        Returns:
            list | None: A list of dictionaries, each representing a review,
                         or None if scraping fails. Returns an empty list
                         if the file already exists and scraping is skipped.
        """
        if app_id not in self.app_ids_map:
            logging.warning(f"App ID '{app_id}' not found in the provided app_ids_map. Skipping.")
            return None # Indicate that this app ID wasn't meant to be processed

        bank_name = self.app_ids_map.get(app_id, 'Unknown_App').replace(' ', '_')
        raw_filename = f'{bank_name}_raw_{self.today_date_str}.csv'
        raw_filepath = os.path.join(self.raw_data_dir, raw_filename)

        logging.info(f"\nScraping reviews for {app_id} ({bank_name})...")

        # Check if a raw data file for this app already exists for today
        if os.path.exists(raw_filepath):
            logging.info(f"Raw data file for {bank_name} already exists today at {raw_filepath}. Skipping scraping.")
            # Indicate that the file exists, so the main process can collect it later
            return [] # Return empty list to signify skipping but not failure

        try:
            results = reviews_all(
                app_id,
                lang=lang,
                country=country,
                sort=sort,
                filter_score_with=None,
                sleep_milliseconds=sleep_milliseconds
            )

            logging.info(f"Scraped {len(results)} reviews for {bank_name}.")
            return results

        except Exception as e:
            logging.error(f"Error scraping reviews for {app_id} ({bank_name}): {e}")
            return None # Indicate scraping failure

    def save_reviews_to_csv(self, reviews, app_id):
        """
        Saves a list of review dictionaries to a CSV file.

        Args:
            reviews (list): A list of review dictionaries.
            app_id (str): The Google Play app ID associated with the reviews.

        Returns:
            str | None: The path of the saved file if successful, None otherwise.
        """
        if reviews is None: # Handle the case where scrape_app_reviews returned None
            logging.warning(f"Cannot save reviews for app ID {app_id}: reviews list is None.")
            return None

        if not reviews: # Handle empty list (either no reviews or skipped)
            logging.info(f"No new reviews to save for app ID: {app_id}.")
            # Return the path of the existing file if it exists, otherwise None
            bank_name = self.app_ids_map.get(app_id, 'Unknown_App').replace(' ', '_')
            raw_filename = f'{bank_name}_raw_{self.today_date_str}.csv'
            raw_filepath = os.path.join(self.raw_data_dir, raw_filename)
            return raw_filepath if os.path.exists(raw_filepath) else None


        bank_name = self.app_ids_map.get(app_id, 'Unknown_App').replace(' ', '_')
        raw_filename = f'{bank_name}_raw_{self.today_date_str}.csv'
        raw_filepath = os.path.join(self.raw_data_dir, raw_filename)

        # Define CSV fieldnames based on the scraped data structure and desired output
        fieldnames = ['review_text', 'rating', 'date', 'app_id', 'source']

        try:
            # [1] using DictWriter and writeheader as in the original code
            with open(raw_filepath, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

                for entry in tqdm(reviews, desc=f"Saving {bank_name}"):
                    # Ensure date is formatted correctly or handle None
                    review_date = entry.get('at')
                    date_str = review_date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(review_date, datetime) else None

                    writer.writerow({
                        'review_text': entry.get('content', ''),
                        'rating': entry.get('score', None),
                        'date': date_str,
                        'app_id': app_id,
                        'source': 'Google Play'
                    })
            logging.info(f"Successfully saved raw reviews for {bank_name} to {raw_filepath}.")
            return raw_filepath

        except Exception as e:
            logging.error(f"Error saving reviews for {bank_name} to CSV: {e}")
            return None

    def scrape_all_apps(self):
        """
        Scrapes reviews for all app IDs provided during initialization.

        Returns:
            list: A list of file paths for the successfully scraped CSV files
                  (including paths to files that were skipped because they already existed).
        """
        logging.info("\n--- Starting Batch Web Scraping ---")
        scraped_files = []

        # Use tqdm for the overall app scraping loop
        for app_id in tqdm(self.app_ids_map.keys(), desc="Scraping Apps"):
            reviews = self.scrape_app_reviews(app_id)

            # If reviews is None, scraping failed for this app, skip saving.
            # If reviews is an empty list [], it means the file already existed and was skipped.
            # In either case (None or []), we still want to attempt to get the filepath
            # if the file *should* exist (was either skipped or successfully saved).
            # If reviews is a list of dictionaries, it was successfully scraped.

            if reviews is not None: # Only proceed if scraping didn't encounter a fatal error
                # Attempt to save the reviews. This method also handles the case
                # where reviews is an empty list (meaning the file existed).
                saved_filepath = self.save_reviews_to_csv(reviews, app_id)
                if saved_filepath:
                    scraped_files.append(saved_filepath)
            # If reviews was None, scraping failed, and we don't add anything to scraped_files.

        logging.info("\n--- Batch Web Scraping Complete ---")
        logging.info(f"Scraped and saved files: {scraped_files}")
        return scraped_files

# This part is for demonstration of how to use the module.
# It should not be executed when the file is imported as a module.
# You would typically put this in your main script or notebook.
if __name__ == "__main__":
    # Example Usage in a main script (e.g., main.py) or the Colab notebook

    # Define constants (these would likely come from a config file or another module)
    APP_ID_TO_BANK_NAME = {
        'com.combanketh.mobilebanking': 'Commercial_Bank_of_Ethiopia',
        'com.boa.boaMobileBanking': 'Bank_of_Abyssinia',
        'com.dashen.dashensuperapp': 'Dashen_Bank_Superapp',
    }
    RAW_DATA_DIR = "data/raw"

    # Initialize the scraper
    # Note: The date string is handled internally by the class
    scraper = GooglePlayScraper(
        app_ids_map=APP_ID_TO_BANK_NAME,
        raw_data_dir=RAW_DATA_DIR
    )

    # Run the scraping process for all apps
    # This method returns the list of file paths for the raw data CSVs
    scraped_files_list = scraper.scrape_all_apps()

    # 'scraped_files_list' now contains the paths to the newly created or existing raw data CSVs.
    # You would typically pass this list to your preprocessing module/function.
    print("\n--- Usage Example Complete ---")
    print("List of raw data files available for preprocessing:")
    print(scraped_files_list)