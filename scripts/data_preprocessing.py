import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler

# Define paths based on the project structure
input_file = "./data/raw_data.csv"  # Raw data file in the data folder
output_file = "./data/cleaned_data.csv"  # Processed data file in the same folder

# Step 1: Load the CSV File
print("Loading raw data...")
df = pd.read_csv(input_file)
print("Data loaded successfully!")
print("Preview of raw data:")
print(df.head())

# Step 2: Handle Missing Data
print("\nHandling missing data...")
df.fillna("N/A", inplace=True)  # Fill missing values with "N/A"
print("Missing data handled.")

# Step 3: Encode Categorical Data
print("\nEncoding categorical data...")
# Example: Encoding Yes/No to 1/0 (adjust column names accordingly)
if "participated" in df.columns:  # Replace 'participated' with the relevant column name
    df["participated"] = df["participated"].apply(lambda x: 1 if x == "Yes" else 0)
print("Categorical data encoded.")

# Step 4: Normalize Numerical Features
print("\nNormalizing numerical features...")
scaler = MinMaxScaler()
# Adjust these columns based on your data
if "hours_studied" in df.columns and "confidence_score" in df.columns:
    df[["hours_studied", "confidence_score"]] = scaler.fit_transform(
        df[["hours_studied", "confidence_score"]]
    )
print("Numerical features normalized.")

# Step 5: Preprocess Text Feedback
print("\nPreprocessing text data...")

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens)

if "feedback" in df.columns:  # Replace 'feedback' with the name of the text column
    df["cleaned_feedback"] = df["feedback"].apply(clean_text)
print("Text data preprocessed.")

# Step 6: Feature Engineering
print("\nPerforming feature engineering...")
if (
    "hours_studied" in df.columns
    and "confidence_score" in df.columns
    and "participated" in df.columns
):
    df["engagement_score"] = (
        df["hours_studied"] * df["confidence_score"] * df["participated"]
    )
print("Feature engineering completed.")

# Step 7: Save the Cleaned Data
print("\nSaving the cleaned data...")
df.to_csv(output_file, index=False)
print(f"Cleaned data saved to '{output_file}'!")

# Preview of cleaned data
print("\nPreview of cleaned data:")
print(df.head())
