import joblib
import os

# Get the directory of the current script (predict.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model, vectorizer, and label binarizer using the correct paths
model_path = os.path.join(current_dir, 'model.pkl')
tfidf_path = os.path.join(current_dir, 'tfidf.pkl')
mlb_path = os.path.join(current_dir, 'mlb.pkl')

# Load the model, vectorizer, and label binarizer
model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)
mlb = joblib.load(mlb_path)

def predict(codes):
    """
    Predict car issues based on car codes.
    
    Args:
    - codes (list): A list of car codes to predict issues for.
    
    Returns:
    - A list of predicted car issues.
    """
    # Convert the car codes to TF-IDF format
    codes_tfidf = tfidf.transform(codes)
    
    # Predict the issues
    predicted_issues_binary = model.predict(codes_tfidf)
    
    # Convert binary output to actual car issues
    predicted_issues = mlb.inverse_transform(predicted_issues_binary)
    
    return predicted_issues
