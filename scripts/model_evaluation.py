from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, precision_recall_curve,average_precision_score
import pandas as pd
import time
import numpy as np
import joblib
import os

def evaluate_xgboost_with_randomsearch(X_train, y_train, X_test, y_test):
    param_grid = {
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [6, 10, 15],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0],
        'xgb__scale_pos_weight': [1, y_train.value_counts()[0] / y_train.value_counts()[1]] 
    }

    # Create a pipeline with SMOTE and XGBoost
    pipe = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])

    # Initialize RandomizedSearchCV with more iterations to ensure thorough search
    # 'cv=3' specifies 3-fold cross-validation
    random_search = RandomizedSearchCV(estimator=pipe, param_distributions=param_grid, cv=3, scoring='roc_auc', n_iter=50, n_jobs=-1, verbose=3, random_state=42)

    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    end_time = time.time()
    


    elapsed_time = end_time - start_time
   
    print(f"Time taken for RandomizedSearchCV: {elapsed_time:.2f} seconds")

    # Get the best model
    best_model = random_search.best_estimator_

   
    print("Best Hyperparameters:", random_search.best_params_)

   
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

  
    print("Training Performance:")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"F1 Score: {f1_score(y_train, y_train_pred):.4f}")
    print(f"Recall Score: {recall_score(y_train, y_train_pred):.4f}")
    print(f"Precision Score: {precision_score(y_train, y_train_pred):.4f}")

  
    print("Testing Performance:")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")
    print(f"Recall Score: {recall_score(y_test, y_test_pred):.4f}")
    print(f"Precision Score: {precision_score(y_test, y_test_pred):.4f}")

   
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f'ROC-AUC: {roc_auc:.4f}')

    # Precision recall data for graphs
    y_score = best_model.predict_proba(X_test)[:, 1]

    # compute precision, recall at all thresholds
    precision, recall,_ = precision_recall_curve(y_test, y_score)
    avg_prec = average_precision_score(y_test, y_score)

    # package into a dict you can jsonify into the template
    pr_data = {
        "precision": precision.tolist(),
        "recall":    recall.tolist(),
        "avg_prec":  round(avg_prec, 2)
    }


    # Save the model
    this_file = os.path.abspath(__file__)
    base_dir  = os.path.dirname(os.path.dirname(this_file))  # go from scripts/ to root
    models_dir = os.path.join(base_dir, 'models')
    model_path = os.path.join(models_dir, 'Fraud_model.joblib')

    joblib.dump(best_model,model_path)
    print(f"Model saved as xgboost_model.pkl at: {model_path}")
    return pr_data


if __name__ == "__main__":
   
    csv_path = os.path.join('app','data','creditcard_sampled.csv')
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Class'])
    y = df['Class']

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply SMOTE to handle class imbalance in the training set
    sm = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

    # Evaluate XGBoost model with RandomizedSearchCV
    evaluate_xgboost_with_randomsearch(X_train_resampled, y_train_resampled, X_test, y_test)


   
