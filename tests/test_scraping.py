import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
from datetime import datetime  # Added missing import

# Add the parent directory to the Python path to import modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from scripts.scraper import GooglePlayScraper  # Fixed import path
from scripts.utils import RAW_DATA_DIR, APP_ID_TO_BANK_NAME, TODAY_DATE_STR, create_directories

# Define dummy data for mocking the scraper
DUMMY_REVIEWS = [
    {'content': 'Great app!', 'score': 5, 'at': datetime(2023, 10, 27, 10, 0, 0)},
    {'content': 'Very slow.', 'score': 1, 'at': datetime(2023, 10, 27, 11, 0, 0)},
    {'content': 'Needs more features.', 'score': 4, 'at': datetime(2023, 10, 27, 12, 0, 0)},
]

class TestScraping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for tests: Create raw data directory."""
        cls.raw_data_dir = RAW_DATA_DIR
        create_directories() # Ensure directories exist

    @classmethod
    def tearDownClass(cls):
        """Tear down after tests: Clean up dummy files."""
        # Optionally clean up the dummy raw files created during tests
        for app_id in APP_ID_TO_BANK_NAME.keys():
             bank_name = APP_ID_TO_BANK_NAME[app_id].replace(' ', '_')
             dummy_filepath = os.path.join(cls.raw_data_dir, f'{bank_name}_raw_{TODAY_DATE_STR}.csv')
             if os.path.exists(dummy_filepath):
                 os.remove(dummy_filepath)

    @patch('scripts.scraper.reviews_all')
    def test_scrape_reviews_saves_file(self, mock_reviews_all):
        """Test that GooglePlayScraper calls reviews_all and saves a CSV."""
        # Configure the mock to return dummy data
        mock_reviews_all.return_value = DUMMY_REVIEWS

        # Define dummy app IDs for this test
        dummy_app_ids = {'com.test.app1': 'Test_App_1'}
        dummy_app_name = 'Test_App_1'

        expected_filepath = os.path.join(self.raw_data_dir, f'{dummy_app_name}_raw_{TODAY_DATE_STR}.csv')

        # Create scraper instance and run
        scraper = GooglePlayScraper(app_ids_map=dummy_app_ids, raw_data_dir=self.raw_data_dir)
        scraped_files = scraper.scrape_all_apps()

        # Assertions
        mock_reviews_all.assert_called_once_with(
            'com.test.app1',
            lang='en',
            country='us',
            sort=MagicMock(), # We don't strictly care about the Sort object instance in mock
            filter_score_with=None,
            sleep_milliseconds=100
        )
        self.assertIn(expected_filepath, scraped_files)
        self.assertTrue(os.path.exists(expected_filepath))

        # Verify content (basic check)
        df = pd.read_csv(expected_filepath)
        self.assertEqual(len(df), len(DUMMY_REVIEWS))
        self.assertListEqual(list(df.columns), ['review_text', 'rating', 'date', 'app_id', 'source'])
        self.assertEqual(df['app_id'][0], 'com.test.app1')
        self.assertEqual(df['rating'][0], DUMMY_REVIEWS[0]['score'])
        self.assertEqual(df['review_text'][0], DUMMY_REVIEWS[0]['content'])

        # Clean up the created file
        if os.path.exists(expected_filepath):
             os.remove(expected_filepath)


    @patch('scripts.scraper.reviews_all')
    def test_scrape_reviews_skips_existing_file(self, mock_reviews_all):
        """Test that GooglePlayScraper skips scraping if a file for today already exists."""
        dummy_app_ids = {'com.test.app2': 'Test_App_2'}
        dummy_app_name = 'Test_App_2'

        expected_filepath = os.path.join(self.raw_data_dir, f'{dummy_app_name}_raw_{TODAY_DATE_STR}.csv')

        # Create a dummy existing file
        with open(expected_filepath, 'w') as f:
            f.write("header\n") # Just needs to exist

        # Create scraper instance and run
        scraper = GooglePlayScraper(app_ids_map=dummy_app_ids, raw_data_dir=self.raw_data_dir)
        scraped_files = scraper.scrape_all_apps()

        # Assertions
        mock_reviews_all.assert_not_called() # Ensure the scraper was not called
        self.assertIn(expected_filepath, scraped_files) # Ensure the existing file is added to the list
        self.assertTrue(os.path.exists(expected_filepath)) # Ensure the dummy file was not removed

        # Clean up
        if os.path.exists(expected_filepath):
             os.remove(expected_filepath)


    @patch('scripts.scraper.reviews_all')
    def test_scrape_reviews_handles_no_results(self, mock_reviews_all):
        """Test that GooglePlayScraper handles the case where no reviews are found."""
        mock_reviews_all.return_value = [] # Mock returning an empty list

        dummy_app_ids = {'com.test.app3': 'Test_App_3'}
        dummy_app_name = 'Test_App_3'

        expected_filepath = os.path.join(self.raw_data_dir, f'{dummy_app_name}_raw_{TODAY_DATE_STR}.csv')

        # Create scraper instance and run
        scraper = GooglePlayScraper(app_ids_map=dummy_app_ids, raw_data_dir=self.raw_data_dir)
        scraped_files = scraper.scrape_all_apps()

        # Assertions
        mock_reviews_all.assert_called_once() # Scraper should be called
        self.assertNotIn(expected_filepath, scraped_files) # No file should be created or added
        self.assertFalse(os.path.exists(expected_filepath)) # No file should exist


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)