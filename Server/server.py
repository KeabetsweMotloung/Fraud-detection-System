import os
import time
import sys
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from threading import Thread
from imblearn.over_sampling import SMOTE



current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, '..','Client','templates')
static_dir = os.path.join(current_dir, '..','Client','static')


current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(current_dir, '..', 'scripts'))
sys.path.append(scripts_dir)

from model_evaluation import evaluate_xgboost_with_randomsearch




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

Upload = 'uploads'
if not os.path.exists(Upload):
    os.makedirs(Upload)

app.config['UPLOAD'] = Upload


# ROUTING
@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/analytics')
def analytics():
    return render_template('Analytics.html')

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

        
        expected_features = model.named_steps['xgb'].feature_names_in_
        df = df[expected_features]

        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[:, 1][0]

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


@app.route("/api/pr-curve")
def pr_curve_api():

    try:
        csv_path = os.path.join(current_dir,'..', 'app','data','creditcard_sampled.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find csv at {csv_path}")
    # load or re-split your data here exactly as in __main__
    
        df = pd.read_csv(csv_path)
        X = df.drop(columns=['Class'])
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42 )
        sm = SMOTE(random_state=42)
        X_res, y_res   = sm.fit_resample(X_train, y_train)

        # this returns the dict you printed
        model_path = os.path.join(current_dir,'..','models','fraud_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")    
        
        model = joblib.load(model_path)
        
        # Generate predictions for PR curve
        y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        avg_prec = average_precision_score(y_test, y_score)


        pr_data = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "avg_prec": round(avg_prec, 2)
        }

        return jsonify(pr_data)


    except Exception as e:
        app.logger.error("Error in /api/pr-curve", exc_info=e)
        return jsonify({"error": str(e)}),500



if __name__ == "__main__":

  app.run(debug=True)
