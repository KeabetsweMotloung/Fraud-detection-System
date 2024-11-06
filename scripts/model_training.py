from data_preprocessing import split_data, load_csv, prepare_data


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import confusion_matrix, f1_score, recall_score,precision_score




def test_models(X_train, y_train, X_test, y_test):
    models={
        'Logistic_Regression': LogisticRegression(),
        'Random_Forest_Classifier': RandomForestClassifier(),
        'KNeighbors_Classifier': KNeighborsClassifier(),
        'SVC': SVC(),
        'Decision_Tree_Classifier': DecisionTreeClassifier(),

    }

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Training set performance
        model_train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
        model_train_f1_score = f1_score(y_train, y_train_pred)
        model_train_recall_score = recall_score(y_train, y_train_pred)
        model_train_recall_precision_score = precision_score(y_train, y_train_pred)

        # Testing set performance
        model_test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
        model_test_f1_score = f1_score(y_test, y_test_pred)
        model_test_recall_score = recall_score(y_test, y_test_pred)
        model_test_recall_precision_score = precision_score(y_test, y_test_pred)

        # Print performance metrics
        print(f"{model_name} Performance:")
        print(f"Training Confusion matrix: {model_train_confusion_matrix}")
        print(f"Training F1 score: {model_train_f1_score:.4f}")
        print(f"Training Recall score: {model_train_recall_score:.4f}")
        print(f"Training Precision score: {model_train_recall_precision_score:.4f}")
        print(f"Testing Confusion matrix: {model_test_confusion_matrix}")
        print(f"Testing F1 score: {model_test_f1_score:.4f}")
        print(f"Testing Recall score: {model_test_recall_score:.4f}")
        print(f"Testing Precision score: {model_test_recall_precision_score:.4f}")
        print("-" * 30)


''' Logistic and Random forest performed the best in training but
 had poor testing performance'''



if __name__ == "__main__":
    csv_path = '//home/keabetswe/Desktop/GitHub/Data Science/Supervised_Learning/Fraud-detection-System/app/data/creditcard_sampled.csv'
    df = load_csv(csv_path)
    X, y = prepare_data(df)
    X_train_resampled, X_test, y_train_resampled, y_test = split_data(X, y,df)

    # Test models and print results
    test_models(X_train_resampled, y_train_resampled, X_test, y_test)