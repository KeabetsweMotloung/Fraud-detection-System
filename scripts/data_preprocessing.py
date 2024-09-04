import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split

csv_path = '/home/keabetswe/Desktop/GitHub/Data Science/Supervised_Learning/Fraud-Detection-Model/app/data/creditcardfraud.zip'


def load_csv(csv_path):
    df = pd.read_csv(csv_path, compression='gzip')
    return df
    
def prepare_data(df):
    '''Split X and Y(target). X is our features(independent variables) 
    and the Y is the target(Dependent variable)'''

    X = df.drop(['isFraud','isFlaggedFraud'], axis = 1)
    y = df['isFraud']
    return X,y



def split_data(X,y):

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

      # Apply NearMiss to the training data
    nearmiss = NearMiss(sampling_strategy='majority')
    X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test


