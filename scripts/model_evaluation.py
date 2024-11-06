from data_preprocessing import split_data, load_csv, prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve

def evaluate_models(X_train, y_train, X_test, y_test):
    # Define hyperparameter grids for tuning
    param_grid_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],               # Type of regularization
        'solver': ['liblinear', 'saga'],        # Solver options
        'class_weight': ['balanced', None]      # Class weight adjustment for imbalanced data
    }

    # Initialize Logistic Regression model
    log_reg = LogisticRegression(max_iter=2000)

    # Stratified Cross-Validation setup to ensure balanced folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV for hyperparameter tuning with Stratified CV
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, cv=cv, scoring='roc_auc')

    # Fit grid search to find the best hyperparameters
    grid_search.fit(X_train, y_train)

    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Display the best hyperparameters
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Training set performance metrics
    model_train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
    model_train_f1_score = f1_score(y_train, y_train_pred)
    model_train_recall_score = recall_score(y_train, y_train_pred)
    model_train_precision_score = precision_score(y_train, y_train_pred)

    # Testing set performance metrics
    model_test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
    model_test_f1_score = f1_score(y_test, y_test_pred)
    model_test_recall_score = recall_score(y_test, y_test_pred)
    model_test_precision_score = precision_score(y_test, y_test_pred)

    # Print performance metrics
    print("Logistic Regression Performance:")
    print(f"Training Confusion Matrix:\n{model_train_confusion_matrix}")
    print(f"Training F1 Score: {model_train_f1_score:.4f}")
    print(f"Training Recall Score: {model_train_recall_score:.4f}")
    print(f"Training Precision Score: {model_train_precision_score:.4f}")
    print(f"Testing Confusion Matrix:\n{model_test_confusion_matrix}")
    print(f"Testing F1 Score: {model_test_f1_score:.4f}")
    print(f"Testing Recall Score: {model_test_recall_score:.4f}")
    print(f"Testing Precision Score: {model_test_precision_score:.4f}")
    print("-" * 30)

    # Evaluate using ROC-AUC
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f'ROC-AUC: {roc_auc:.4f}')

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f"Precision-Recall curve generated.")

if __name__ == "__main__":
    csv_path = '/home/keabetswe/Desktop/GitHub/Data Science/Supervised_Learning/Fraud-detection-System/app/data/creditcard_sampled.csv'
    df = load_csv(csv_path)
    X, y = prepare_data(df)
    X_train_resampled, X_test, y_train_resampled, y_test = split_data(X, y, df)

    # Test models and print results
    evaluate_models(X_train_resampled, y_train_resampled, X_test, y_test)
