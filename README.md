from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load('medical_recommendation_model.pkl')

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get input data from the request
    input_data = request.json
    # Preprocess input data
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=features, fill_value=0)
    input_scaled = scaler.transform(input_df)
    # Make a prediction
    prediction = model.predict(input_scaled)
