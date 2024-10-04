# this is a flask api that runs on 8080 at 0.0.0.0 we need to load both vectors and model to make predictions
# we will use the predict route to make predictions

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Load the model and vectorizer
model = joblib.load('perceptron_model_v1.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Ensure you save the vectorizer as well during training

# Initialize the Flask application
app = Flask(__name__)

# Define a route to predict the fall flag
@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the incident description from the request
    description = request.json['description']

    # Vectorize the input
    vectorized_input = vectorizer.transform([description])

    # Make a prediction
    prediction = model.predict(vectorized_input)[0]

    return jsonify({'prediction': int(prediction)})

# Run the Flask application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
