{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Customer Experience Analytics for Banking Apps\n",
    "## Data Preprocessing and EDA"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Install Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages quietly\n",
    "import sys\n",
    "import subprocess\n",
    "\n",
    "def quiet_install(package):\n",
    "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"--quiet\", package])\n",
    "\n",
    "quiet_install('pandas')\n",
    "quiet_install('seaborn')\n",
    "quiet_install('matplotlib')\n",
    "quiet_install('tqdm')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from tqdm import tqdm\n",
    "\n",
    "# Set plotting style\n",
    "plt.style.use('seaborn')\n",
    "sns.set_palette('pastel')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Set Up Paths"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add scripts directory to path\n",
    "scripts_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'scripts'))\n",
    "if scripts_path not in sys.path:\n",
    "    sys.path.append(scripts_path)\n",
    "\n",
    "# Define data directories\n",
    "RAW_DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'raw'))\n",
    "CLEANED_DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'cleaned'))\n",
    "os.makedirs(CLEANED_DATA_DIR, exist_ok=True)\n",
    "\n",
    "# Bank name mapping\n",
    "APP_ID_TO_BANK_NAME = {\n",
    "    'com.example.bank1': 'Bank A',\n",
    "    'com.example.bank2': 'Bank B',\n",
    "    'com.example.bank3': 'Bank C'\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import preprocessing module\n",
    "from preprocessing import ReviewPreprocessor\n",
    "\n",
    "# Get list of raw files\n",
    "raw_files = [os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]\n",
    "print(f\"Found {len(raw_files)} raw data files\")\n",
    "\n",
    "# Initialize and run preprocessor\n",
    "preprocessor = ReviewPreprocessor(\n",
    "    cleaned_data_dir=CLEANED_DATA_DIR,\n",
    "    app_id_to_bank_name=APP_ID_TO_BANK_NAME\n",
    ")\n",
    "\n",
    "combined_df = preprocessor.preprocess_batch(raw_files)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Load Cleaned Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the cleaned data\n",
    "combined_cleaned_path = os.path.join(CLEANED_DATA_DIR, 'all_reviews_cleaned.csv')\n",
    "df = pd.read_csv(combined_cleaned_path)\n",
    "\n",
    "# Data cleaning\n",
    "df = df.dropna(subset=['review_text'])  # Drop rows with missing review text\n",
    "df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')  # Convert to datetime\n",
    "df['year_month'] = df['review_date'].dt.to_period('M')  # Create year-month column\n",
    "\n",
    "# Preview data\n",
    "print(f\"Data shape: {df.shape}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.1 Data Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\nData Summary:\")\n",
    "print(df.info())\n",
    "\n",
    "print(\"\\nMissing Values:\")\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 Rating Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(data=df, x='rating')\n",
    "plt.title('Distribution of Ratings')\n",
    "plt.xlabel('Rating (1-5)')\n",
    "plt.ylabel('Count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.3 Reviews per Bank"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(data=df, y='bank_name', order=df['bank_name'].value_counts().index)\n",
    "plt.title('Number of Reviews per Bank')\n",
    "plt.xlabel('Count')\n",
    "plt.ylabel('Bank Name')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.4 Review Length Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['review_length'] = df['review_text'].astype(str).apply(len)\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.histplot(df['review_length'], bins=50)\n",
    "plt.title('Distribution of Review Lengths')\n",
    "plt.xlabel('Review Length (characters)')\n",
    "plt.ylabel('Count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.5 Ratings Over Time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 8))\n",
    "sns.countplot(data=df, x='year_month', hue='bank_name')\n",
    "plt.title('Reviews Over Time by Bank')\n",
    "plt.xlabel('Year-Month')\n",
    "plt.ylabel('Count')\n",
    "plt.xticks(rotation=45)\n",
    "plt.legend(title='Bank Name')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.6 Average Rating by Bank"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "avg_ratings = df.groupby('bank_name')['rating'].mean().sort_values(ascending=False)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=avg_ratings.values, y=avg_ratings.index)\n",
    "plt.title('Average Rating by Bank')\n",
    "plt.xlabel('Average Rating')\n",
    "plt.ylabel('Bank Name')\n",
    "plt.xlim(0, 5)  # Ratings are 1-5\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Save Analysis Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save processed data with additional features\n",
    "output_path = os.path.join(CLEANED_DATA_DIR, 'analyzed_reviews.csv')\n",
    "df.to_csv(output_path, index=False)\n",
    "print(f\"Analysis results saved to {output_path}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}