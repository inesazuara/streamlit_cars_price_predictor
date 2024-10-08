import pandas as pd
import streamlit as st
import pickle
from model import load_data,preprocess_data,load_model
import numpy as np

def collect_car_info(df_cars):
    st.markdown("<h5>Please, introduce your car properties</h5>", unsafe_allow_html=True)
    # Collecting car information
    unique_make = df_cars['make'].unique()
    make_selected = st.selectbox("Select your make",unique_make)

    models_filtered = df_cars[df_cars['make'] == make_selected]['model'].unique()

    model_selected = st.selectbox("Select you model",models_filtered)

    months_old_selected = st.number_input("Introduce the car's age in months",min_value=0,max_value=1000)

    power_selected = st.number_input("Introduce the car's power",min_value=0,max_value=1000)

    sale_type_selected = st.selectbox("Select the car's condition",df_cars['sale_type'].unique())

    num_owners_selected = st.number_input("Select the car's condition",min_value=1,max_value=20,step=1)
    unknown_owners = st.checkbox("I don't know the numer of owners")
    if unknown_owners:
        num_owners = np.nan

    gear_type_selected = st.selectbox("Select the car's gear type",df_cars['gear_type'].unique())
    fuel_type_selected = st.selectbox("Select the car's fuel type",df_cars['fuel_type'].unique())

    kms_selected = st.number_input("Introduce the car's kilometers",min_value=0,max_value=3000000)

    return {
        'make': make_selected,
        'model': model_selected,
        'months_old': months_old_selected,
        'power': power_selected,
        'sale_type': sale_type_selected,
        'num_owners': num_owners_selected,
        'gear_type': gear_type_selected,
        'fuel_type': fuel_type_selected,
        'kms': kms_selected,
        'price': 0 # Dummy for fitting model columns
    }

def predict_car_price(model, car_info):
    car_df = pd.DataFrame([car_info])
    car_df_processed = preprocess_data(car_df)
    #Loading expected columns to fit the model
    with open('models/expected_columns.pkl', 'rb') as file:
        expected_columns = pickle.load(file)

    # Be sure that columns match with the training model
    missing_cols = [col for col in expected_columns if col not in car_df_processed.columns]
    for col in missing_cols:
        car_df_processed[col] = 0

    car_df_processed = car_df_processed[expected_columns]
    # Checking if price columns exists, if does, drop it
    if 'price' in car_df_processed.columns:
        car_df_processed = car_df_processed.drop('price', axis=1)


    predicted_price = model.predict(car_df_processed)
    return predicted_price[0]

if __name__ == "__main__":
    st.title('Price car predictor')
    st.markdown('This predictor has been trainned with data from spanish car sells during years 2014-2016.  \n'
                'It uses an AdaBoost model.')

    # Loading dataset and model
    df_cars = load_data("data/coches_usados_esp.csv",";")
    car_info = collect_car_info(df_cars)
    model = load_model("models/model_AdaBoost.pkl")

    # Executing the model when user press the button
    if st.button("Predict Price"):
        predicted_price = predict_car_price(model,car_info)

        formatted_price = "{:,.2f}".format(predicted_price).replace(",", "X").replace(".", ",").replace("X", ".")
        st.write(f"The predicted price for your car is: {formatted_price} €")

