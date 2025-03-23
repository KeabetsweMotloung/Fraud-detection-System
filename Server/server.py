import os
import time
import sys
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from threading import Thread
from imblearn.over_sampling import SMOTE


current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, '..','Client','templates')
static_dir = os.path.join(current_dir, '..','Client','static')



app = Flask(__name__,static_folder=static_dir, template_folder=template_dir)
CORS(app)

# Load the saved model at startup
model_path = os.path.join(current_dir,'..', 'models', 'Fraud_model.joblib')

# Check if the model file exists before loading it
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
else:
    print(f"Model file not found at: {model_path}")
    sys.exit(1)  



# ROUTING
@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/dashboard')
def dashboard():
    return render_template('Dashboard.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the model only when needed
        model = joblib.load(model_path)
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure input matches model's expected features
        expected_features = model.named_steps['xgb'].feature_names_in_
        df = df[expected_features]

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[:, 1][0]

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":

  app.run(debug=True)
