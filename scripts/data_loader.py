import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile

def csv_file():
    df = pd.read_csv("app/data/creditcard_sampled.csv")
    
    # Display the first few rows and column names to understand the structure
    print(df.head())
    print(df.columns)
    print(df.shape)

    return df

if __name__ == "__main__":
    csv_file()

# def sample_dataset(df, sample_fraction=0.3, stratify=True):
#     if stratify:
#         # Stratified sampling based on the 'Class' column
#         df_sampled, _ = train_test_split(df, test_size=(1 - sample_fraction), stratify=df['Class'], random_state=42)
#     else:
#         # Random sampling without stratification
#         df_sampled = df.sample(frac=sample_fraction, random_state=42)

#     return df_sampled

# if __name__ == "__main__":
#     df = csv_file()
#     df_sampled = sample_dataset(df, sample_fraction=0.3, stratify=True)
    
#     # Save the sampled dataset to a new CSV file
#     df_sampled.to_csv('app/data/creditcard_sampled.csv', index=False)
    
#     print(f"Sampled dataset shape: {df_sampled.shape}")
