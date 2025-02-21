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
# scripts_path = '/home/keabetswe/Desktop/GitHub/Data Science/Supervised_Learning/Fraud-detection-System/scripts'
# if scripts_path not in sys.path:
#     sys.path.append(scripts_path)


# from model_evaluation import evaluate_xgboost_with_randomsearch 

# Determine the absolute paths to template and static folders
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, '../Client/templates')
static_dir = os.path.join(current_dir, '../Client/static')
# static_dir = os.path.join(current_dir, '../Client/frontend/build')



app = Flask(__name__, static_folder='../Client/static', template_folder='../Client/templates')
# app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Load the saved model at startup
model_path = "/home/keabetswe/Desktop/GitHub/Data Science/Supervised_Learning/Fraud-detection-System/models/Fraud_model.joblib"
model = joblib.load(model_path)




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

        # Get JSON data from request
        data = request.get_json()

        # Convert JSON to DataFrame
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
