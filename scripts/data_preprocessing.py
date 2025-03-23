import pandas as pd
import os
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split

csv_path = os.path.join("app","data","creditcard_sampled.csv")


def load_csv(csv_path):
    df = pd.read_csv(csv_path) 
    return df

# Prepare data by splitting into features (X) and target (y)
def prepare_data(df):
    X = df.drop(['Class'], axis=1)  
    y = df['Class']  
    return X, y

# Split data into training and test sets, then apply NearMiss to balance the classes
def split_data(X, y, df):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a 50/50 sub-dataframe ratio of "Fraud" and "Non-Fraud" transactions (NearMiss Algorithm)
    nearmiss = NearMiss(sampling_strategy=1.0)  # 1.0 means balance to a 50/50 ratio
    X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train, y_train)


    print("Resampled training data class distribution:")
    print(y_train_resampled.value_counts())

    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    df = load_csv(csv_path)
    X, y = prepare_data(df)
    split_data(X, y, df)
