# Car Price Predictor

This project is a web application built with Streamlit that predicts the price of a car based on various input parameters provided by the user. The application uses a machine learning model trained on historical car sales data from Spain (2014-2016) and leverages an AdaBoost regressor to make predictions.

## Features
- Select car brand and model
- Input car attributes such as mileage, power, number of owners, and age in months
- Predict car price based on input parameters
- User-friendly interface built with Streamlit

## Model
The machine learning model behind this application is an **AdaBoost Regressor**, which was trained on data from Spanish car sales between 2014 and 2016. The model was trained using features such as:
- Car brand and model
- Car condition (new, used, etc.)
- Fuel type
- Gear type
- Number of previous owners
- Mileage
- Power (in horsepower)
- Car age in months

## Installation

Follow these steps to set up and run the application on your local machine.

### Prerequisites

Make sure you have Python 3.8 or higher installed on your system. You can check the Python version by running:

```bash
python --version
```
### Clone the repository

```bash
git clone https://github.com/inesazuara/streamlit_cars_price_predictor
cd streamlit_data_exploration
```

### Set up virtual environment (optional but recommendeed)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (MacOS/Linux)
source venv/bin/activate
```

### Install the dependencies

```bash
pip install -r requirements.txt
```

### Usage

If you want to see the deployed version of the app go to: https://app-data-exploration.streamlit.app/

For using the code in your own computer:

```bash
streamlit run app.py
```
This will open the application in your web browser. You can then input various car details, such as the brand, model, mileage, power, and number of owners, and get an estimated car price prediction.
How to use:

- Input car details: Enter information such as car brand, model, age in months, power, number of owners, fuel type, gear type, and kilometers.
- Price prediction: After entering the data, click the Predict Price button to get an estimated price for the car.
- Real-time results: The predicted price will be displayed dynamically based on the car's attributes.