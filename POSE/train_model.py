import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def extract_codes_x(file_path):
    """
    This function extracts the car codes (X values) from the first column of an Excel file.
    
    Args:
    - file_path (str): Path to the Excel file containing car codes.
    
    Returns:
    - A list of car codes (X values).
    """
    df = pd.read_excel(file_path)
    codes = df.iloc[:, 0].tolist()  # Extract only the first column
    return codes

def extract_issues_y(file_path):
    """
    This function extracts car issues (Y values) from an Excel file, skipping the first row (metadata).
    Each row in the Excel file contains a list of issues, one issue per cell.
    
    Args:
    - file_path (str): Path to the Excel file containing car issues.
    
    Returns:
    - A list of lists where each inner list contains the car issues for a given code.
    """
    df = pd.read_excel(file_path, header=None, skiprows=1)  # Skip the first row (metadata)
    
    # Fill missing values (NaN) with empty strings and ensure all values are treated as strings
    df = df.fillna("").astype(str)
    
    issues = df.values.tolist()  # Convert DataFrame rows to a list of lists
    return issues

def create_training_data(codes_file, issues_file):
    """
    This function combines the car codes (X values) and car issues (Y values) into a training dataset.
    
    Args:
    - codes_file (str): Path to the Excel file containing car codes (X).
    - issues_file (str): Path to the Excel file containing car issues (Y).
    
    Returns:
    - A tuple (X, Y) where X is a list of car codes and Y is a list of car issues.
    """
    X = extract_codes_x(codes_file)
    Y = extract_issues_y(issues_file)
    
    # Check if X and Y lengths are unequal
    if len(X) != len(Y):
        print(f"Length of X: {len(X)}")
        print(f"Length of Y: {len(Y)}")
        raise ValueError("The number of car codes (X) and car issues (Y) do not match.")
    
    return X, Y

def train_and_save_model(X, Y):
    """
    This function trains a RandomForestClassifier wrapped in a MultiOutputClassifier on the car codes (X) and car issues (Y),
    and then saves the model, TF-IDF vectorizer, and MultiLabelBinarizer.
    
    Args:
    - X (list): A list of car codes (strings).
    - Y (list): A list of lists where each inner list contains the car issues for a given code.
    """
    # Convert car issues to binary labels using MultiLabelBinarizer
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Y_binary = mlb.fit_transform(Y)  # Transforms Y into a binary matrix format for multi-label classification
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_binary, test_size=0.2, random_state=42)

    # Create a pipeline: TfidfVectorizer to convert text to numbers + RandomForestClassifier
    tfidf = TfidfVectorizer()

    # Transform X data to TF-IDF format
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Use MultiOutputClassifier to handle multi-label classification
    model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    
    # Train the model on the training data
    model.fit(X_train_tfidf, Y_train)
    
    # Evaluate the model on the test set
    Y_pred = model.predict(X_test_tfidf)
    print(classification_report(Y_test, Y_pred, target_names=mlb.classes_))

    # Save the model, vectorizer, and label binarizer
    joblib.dump(model, 'model.pkl')  # Save the trained model
    joblib.dump(tfidf, 'tfidf.pkl')  # Save the TF-IDF vectorizer
    joblib.dump(mlb, 'mlb.pkl')      # Save the MultiLabelBinarizer

    print("Model, TF-IDF vectorizer, and MultiLabelBinarizer have been saved.")

import random

def load_and_predict(sample_codes):
    """
    Load the saved model, TF-IDF vectorizer, and MultiLabelBinarizer and predict the car issues.
    Then, return a random value from the first list and a random one from the second list of issues.
    
    Args:
    - sample_codes (list): A list of car codes to predict issues for.
    
    Returns:
    - A tuple with two random issues: one from the first list of predictions, and one from the second.
    """
    # Load the model, vectorizer, and label binarizer
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    mlb = joblib.load('mlb.pkl')
    
    # Convert the car codes to TF-IDF format
    sample_codes_tfidf = tfidf.transform(sample_codes)
    
    # Predict the issues
    predicted_issues_binary = model.predict(sample_codes_tfidf)
    
    # Convert binary output to actual car issues
    predicted_issues = mlb.inverse_transform(predicted_issues_binary)
    
    # Ensure there are at least two lists of issues to select from
    if len(predicted_issues) < 2:
        return "Not enough data to choose from two lists."
    
    # Randomly select one issue from the first list and one from the second
    random_issue_1 = random.choice(predicted_issues[0]) if predicted_issues[0] else "No issue"
    random_issue_2 = random.choice(predicted_issues[1]) if predicted_issues[1] else "No issue"
    
    return random_issue_1, random_issue_2

# Example usage
if __name__ == "__main__":
    codes_file = 'codes.xlsx'  # Path to the Excel file with car codes
    issues_file = 'output.xlsx'  # Path to the Excel file with car issues
    
    # Create training data
    try:
        X, Y = create_training_data(codes_file, issues_file)
    except ValueError as e:
        print(e)
    
    # Train the model and save it if data lengths match
    if len(X) == len(Y):
        train_and_save_model(X, Y)

        # Example of making a prediction after loading the model
        sample_codes = ['P0001', 'P0002']  # Replace with actual car codes
        predicted_issues = load_and_predict(sample_codes)
        print("Predicted car issues:", predicted_issues)
