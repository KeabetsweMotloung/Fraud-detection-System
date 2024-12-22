from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS
import os

# Determine the absolute paths to template and static folders
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, '../Client/templates')
static_dir = os.path.join(current_dir, '../Client/static')
# static_dir = os.path.join(current_dir, '../Client/frontend/build')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Load the model
model_path = os.path.join(current_dir, '..', 'models', 'Fraud_model.joblib')
medical_model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('Home.html')



@app.route('/predict', methods=["POST"])
def predict():
    data = request.get_json(force=True)
    prediction = medical_model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

@app.route('/stats', methods=['GET'])
def stats():
    csv_path = os.path.join(current_dir, '../app/data/creditcard_sampled.csv')
    df = pd.read_csv(csv_path)
    
    total_transactions = len(df)
    fraud_transactions = df['Class'].sum()
    non_fraud_transactions = total_transactions - fraud_transactions
    
    stats = {
        'total_transactions': total_transactions,
        'fraud_transactions': fraud_transactions,
        'non_fraud_transactions': non_fraud_transactions,
    }
    
    return jsonify(stats)

if __name__ == "__main__":
    app.run(debug=True)
