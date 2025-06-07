import os
from datetime import datetime
import json
import string # Import string module for printable characters

# Define constants
APP_ID_TO_BANK_NAME = {
    'com.combanketh.mobilebanking': 'Commercial_Bank_of_Ethiopia',
    'com.boa.boaMobileBanking': 'Bank_of_Abyssinia',
    'com.dashen.dashensuperapp': 'Dashen_Bank_Superapp',
}

APP_IDS = list(APP_ID_TO_BANK_NAME.keys())

TODAY_DATE_STR = datetime.now().strftime('%Y%m%d')
RAW_DATA_DIR = "data/raw"
CLEANED_DATA_DIR = "data/cleaned"
ANALYSIS_DATA_DIR = "data/analysis"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# --- Oracle Database Constants ---
# **WARNING:** Replace with your actual credentials.
ORACLE_DB_USER = 'your_db_user'
ORACLE_DB_PASSWORD = 'your_db_password'
ORACLE_DB_DSN = 'your_host:your_port/your_service_name'
ORACLE_TABLE_NAME = 'APP_REVIEWS' # Define your table name

# --- Thematic Analysis Keywords (Refined based on scenarios) ---
THEME_KEYWORDS = {
    'Login/Account Issues': ['login', 'account', 'password', 'otp', 'access', 'lock', 'register', 'signup', 'issue login', 'locked', 'failed login', 'verify', 'verification'],
    'Transaction Problems': ['transfer', 'send money', 'receive money', 'payment', 'transaction', 'failed transaction', 'not received', 'delay', 'pending', 'complete'],
    'App Performance (Slow/Crash)': ['slow', 'delay', 'fast', 'speed', 'loading', 'hang', 'crash', 'freeze', 'bug', 'lagging', 'error', 'app stop'],
    'UI/Usability': ['ui', 'design', 'easy use', 'user friendly', 'interface', 'look', 'layout', 'simple', 'confusing', 'difficult navigate', 'intuitive'],
    'Customer Support': ['support', 'customer service', 'help', 'contact', 'response', 'agent', 'solve', 'issue solve', 'call center'],
    'Missing Features/Suggestions': ['feature', 'add', 'option', 'suggestion', 'request', 'allow', 'need', 'fingerprint', 'face id', 'more features'],
    'Updates/Installation': ['update', 'install', 'version', 'download', 'new update'],
    'Network/Connectivity': ['network', 'internet', 'connection', 'offline', 'data']
}

# Define the set of printable characters
PRINTABLE_CHARS = set(string.printable)

def remove_non_printable(text):
    """Removes non-printable characters from a string."""
    if isinstance(text, str):
        return ''.join(filter(lambda x: x in PRINTABLE_CHARS, text))
    return text # Return non-string input as is


def create_directories():
    """Creates necessary directories for the project."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Project directories created/ensured.")

def download_nltk_data():
    """Downloads required NLTK data."""
    import nltk
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}') if resource == 'punkt' else nltk.data.find(f'corpora/{resource}')
            print(f"NLTK '{resource}' already found.")
        except LookupError:
            print(f"NLTK '{resource}' not found. Downloading...")
            nltk.download(resource, quiet=True)
            print(f"NLTK '{resource}' downloaded.")
        except Exception as e:
             print(f"An unexpected error occurred checking/downloading NLTK '{resource}': {e}")


def download_spacy_model():
    """Downloads required spaCy model."""
    import spacy
    import subprocess
    try:
        spacy.load('en_core_web_sm')
        print("spaCy 'en_core_web_sm' model already installed and loaded.")
    except OSError:
        print("Downloading spaCy 'en_core_web_sm' model...")
        try:
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm', '--quiet'], check=True)
            print("spaCy 'en_core_web_sm' model downloaded.")
        except subprocess.CalledProcessError as e:
            print(f"Error during spaCy model download: {e}")