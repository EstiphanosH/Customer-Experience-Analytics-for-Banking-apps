#!/usr/bin/env python3
"""
Main execution script for Customer Experience Analytics for Banking Apps.

This script orchestrates the entire pipeline:
1. Data scraping from Google Play Store
2. Data preprocessing and cleaning
3. Sentiment and thematic analysis
4. Visualization and reporting

Usage:
    python run.py [--skip-scraping] [--skip-analysis] [--skip-plots]
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add scripts directory to Python path
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, scripts_dir)

# Import project modules
from scripts.scraper import GooglePlayScraper
from scripts.preprocessing import ReviewPreprocessor
from scripts.analysis import analyze_reviews, generate_scenario_1_plots, generate_scenario_2_plots, generate_scenario_3_plots, generate_general_eda_plots
from scripts.utils import (
    APP_ID_TO_BANK_NAME, 
    RAW_DATA_DIR, 
    CLEANED_DATA_DIR, 
    FIGURES_DIR,
    TODAY_DATE_STR,
    create_directories,
    download_nltk_data,
    download_spacy_model
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'banking_analytics_{TODAY_DATE_STR}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_environment():
    """Set up the environment and download required models."""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    create_directories()
    
    # Download required NLTK data
    try:
        download_nltk_data()
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Failed to download NLTK data: {e}")
    
    # Download spaCy model
    try:
        download_spacy_model()
        print("✅ spaCy model downloaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Failed to download spaCy model: {e}")

def run_scraping():
    """Execute the web scraping phase."""
    print("\n📱 Starting web scraping phase...")
    
    try:
        # Initialize scraper
        scraper = GooglePlayScraper(
            app_ids_map=APP_ID_TO_BANK_NAME,
            raw_data_dir=RAW_DATA_DIR
        )
        
        # Scrape all apps
        scraped_files = scraper.scrape_all_apps()
        
        if scraped_files:
            print(f"✅ Scraping completed. Files created: {len(scraped_files)}")
            for file_path in scraped_files:
                print(f"   📄 {os.path.basename(file_path)}")
            return scraped_files
        else:
            print("⚠️ No files were scraped. Check your internet connection and app IDs.")
            return []
            
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        logging.error(f"Scraping failed: {e}")
        return []

def run_preprocessing(raw_files):
    """Execute the data preprocessing phase."""
    print("\n🧹 Starting data preprocessing phase...")
    
    if not raw_files:
        print("⚠️ No raw files to process. Skipping preprocessing.")
        return None
    
    try:
        # Initialize preprocessor
        preprocessor = ReviewPreprocessor()
        
        # Process all files
        combined_df = preprocessor.preprocess_batch(raw_files)
        
        if not combined_df.empty:
            print(f"✅ Preprocessing completed. Total reviews: {len(combined_df)}")
            print(f"   📊 Shape: {combined_df.shape}")
            return combined_df
        else:
            print("⚠️ No data remained after preprocessing.")
            return None
            
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        logging.error(f"Preprocessing failed: {e}")
        return None

def run_analysis(cleaned_df):
    """Execute the sentiment and thematic analysis phase."""
    print("\n🧠 Starting analysis phase...")
    
    if cleaned_df is None or cleaned_df.empty:
        print("⚠️ No cleaned data available for analysis.")
        return None
    
    try:
        # Run sentiment and thematic analysis
        analysis_df = analyze_reviews(cleaned_df)
        
        if not analysis_df.empty:
            print(f"✅ Analysis completed. Analyzed reviews: {len(analysis_df)}")
            
            # Show summary statistics
            if 'sentiment_label' in analysis_df.columns:
                sentiment_counts = analysis_df['sentiment_label'].value_counts()
                print(f"   📈 Sentiment distribution: {sentiment_counts.to_dict()}")
            
            if 'identified_themes' in analysis_df.columns:
                theme_counts = analysis_df['identified_themes'].apply(len).describe()
                print(f"   🏷️ Average themes per review: {theme_counts['mean']:.2f}")
            
            return analysis_df
        else:
            print("⚠️ Analysis produced no results.")
            return None
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logging.error(f"Analysis failed: {e}")
        return None

def run_visualization(analysis_df):
    """Execute the visualization phase."""
    print("\n📊 Starting visualization phase...")
    
    if analysis_df is None or analysis_df.empty:
        print("⚠️ No analysis data available for visualization.")
        return
    
    try:
        banks = list(analysis_df['bank_name'].unique()) if 'bank_name' in analysis_df.columns else []
        
        # Generate scenario-specific plots
        print("   📈 Generating Scenario 1 plots (User Retention)...")
        generate_scenario_1_plots(analysis_df, banks)
        
        print("   📈 Generating Scenario 2 plots (Feature Enhancement)...")
        generate_scenario_2_plots(analysis_df)
        
        print("   📈 Generating Scenario 3 plots (Complaint Management)...")
        generate_scenario_3_plots(analysis_df)
        
        print("   📈 Generating general EDA plots...")
        generate_general_eda_plots(analysis_df)
        
        print(f"✅ Visualization completed. Check {FIGURES_DIR} for plots.")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        logging.error(f"Visualization failed: {e}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run Customer Experience Analytics Pipeline')
    parser.add_argument('--skip-scraping', action='store_true', help='Skip the scraping phase')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip the analysis phase')
    parser.add_argument('--skip-plots', action='store_true', help='Skip the plotting phase')
    
    args = parser.parse_args()
    
    print("🏦 Customer Experience Analytics for Ethiopian Banking Apps")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Phase 1: Web Scraping
    if not args.skip_scraping:
        raw_files = run_scraping()
    else:
        print("\n📱 Skipping scraping phase...")
        # Look for existing raw files
        raw_files = []
        if os.path.exists(RAW_DATA_DIR):
            raw_files = [
                os.path.join(RAW_DATA_DIR, f) 
                for f in os.listdir(RAW_DATA_DIR) 
                if f.endswith('.csv')
            ]
        print(f"   📄 Found {len(raw_files)} existing raw files")
    
    # Phase 2: Data Preprocessing
    cleaned_df = run_preprocessing(raw_files)
    
    # Phase 3: Analysis
    if not args.skip_analysis:
        analysis_df = run_analysis(cleaned_df)
    else:
        print("\n🧠 Skipping analysis phase...")
        analysis_df = cleaned_df
    
    # Phase 4: Visualization
    if not args.skip_plots:
        run_visualization(analysis_df)
    else:
        print("\n📊 Skipping visualization phase...")
    
    print("\n🎉 Pipeline execution completed!")
    print(f"📋 Check the log file: banking_analytics_{TODAY_DATE_STR}.log")
    print(f"📊 Check visualizations in: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
