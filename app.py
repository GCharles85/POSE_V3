import logging
from flask import Flask, render_template, request, jsonify
from POSE import predict
import joblib

app = Flask(__name__)

# Set up logging to the console (stdout)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict_issues():
    try:
        # Ensure request content type is correct
        if request.content_type != 'application/json':
            return jsonify({'error': 'Invalid content type, expected JSON'}), 400 
        
        # Get the JSON body
        json_body = request.get_json()
        if not json_body or 'codes' not in json_body:
            return jsonify({'error': 'No codes selected!'}), 400

        selected_codes = json_body['codes']
        
        # Assuming predict is a valid function
        predicted_issues = predict.predict(selected_codes)

        # Return predictions as JSON response
        return jsonify({
            "selected_codes": selected_codes,
            "predicted_issues": predicted_issues
        })
    
    except Exception as e:
        # Catch any other exceptions and return a valid JSON response
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
