# Importing libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import root_mean_squared_error
import numpy as np


# Loading dataset for training
def load_data(file_path,sep):
       return pd.read_csv(file_path, sep=sep)

# print(df_cars.head())

def preprocess_data(df):
       # Selecting variables for the model and diveding by numeric and categorical
       cols = ['make', 'model', 'months_old', 'power', 'sale_type',
              'num_owners', 'gear_type', 'fuel_type', 'kms', 'price']

       cat_cols = ['make', 'model', 'sale_type', 'gear_type', 'fuel_type']

       # Creating dummies
       df = pd.get_dummies(df[cols],
                           prefix_sep='_',
                           drop_first=True,
                           columns=cat_cols)

       # Cleaning nulls
       # First cleaning num_owners nulls converting in categorical variable and creating Null category
       df['num_owners'] = df['num_owners'].astype('object')
       filter_num_owner = df['num_owners'] >= 3
       df.loc[filter_num_owner,'num_owners'] = '3+'

       df = pd.get_dummies(df,prefix_sep='_',
                           dummy_na=True,
                           drop_first=True,
                           columns=['num_owners'])

       # Assigning median to the rest of the columns with null values
       for col in ['months_old', 'power', 'kms']:
           df[col] = df[col].fillna(df[col].median())

       return df


def train_model(data):
    X = data.drop('price',axis=1)
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = ensemble.AdaBoostRegressor()

    # Train the model
    model.fit(X_train, y_train)
    return model, X_test, X_train, y_test, y_train

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculating RMSE for train and test
    rmse_train = round(root_mean_squared_error(y_train, y_train_pred), 2)
    rmse_test = round(root_mean_squared_error(y_test, y_test_pred), 2)

    # Degradation
    degradation = round((rmse_test - rmse_train)/rmse_train*100,2)

    # Printing results
    print(f"RMSE (Train): {rmse_train}")
    print(f"RMSE (Test): {rmse_test}")
    print(f"Degradation: {degradation}%")

    return rmse_train, rmse_test, degradation

def save_model(model,file_name):
    # Saving model to an .pkl file
    with open(file_name, 'wb') as file:
        pickle.dump(model,file)

def load_model(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
       df_cars = load_data("data/coches_usados_esp.csv",";")
       df_cars_clean = preprocess_data(df_cars)

       # Model
       model, X_train, X_test, y_train, y_test = train_model(df_cars_clean)
       evaluate_model(model, X_train, X_test, y_train, y_test)
       save_model(model,"models/model_AdaBoost.pkl")
       print("Model saved has models/model_AdaBoost.pkl")

