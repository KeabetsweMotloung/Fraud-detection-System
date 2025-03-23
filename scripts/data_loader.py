import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile
import os

def csv_file():
    csv_path = os.path.join("app","data","creditcard_sampled.csv")
    df = pd.read_csv(csv_path)
    
    
    print(df.head())
    print(df.columns)
    print(df.shape)

    return df

if __name__ == "__main__":
    csv_file()



