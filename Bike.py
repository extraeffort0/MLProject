import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to preprocess data
def preprocess_data(data):
    # Identify non-numeric columns
    for col in data.select_dtypes(include=['object', 'category']).columns:
        # Try parsing dates or encoding categorical variables
        try:
            data[col] = pd.to_datetime(data[col]).dt.year  # Convert dates to years
        except ValueError:
            data = pd.get_dummies(data, columns=[col], drop_first=True)  # One-hot encode

    # Drop irrelevant columns, if any
    try:
        return data.drop(columns=['cnt', 'atemp', 'registered'], axis=1)
    except KeyError as e:
        st.error(f"Missing columns in the data: {e}")
        return None

# Function to train the model
def train_model(data):
    y = data['cnt']
    X = preprocess_data(data)
    if X is not None:
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        rmse = calculate_rmse(y, y_pred)
        return model, r_squared, rmse
    else:
        return None, None, None

# Main App
def main():
    st.title("Model Performance Metrics")

    # Train data upload
    st.header("Upload Training Data")
    train_file = st.file_uploader("Upload a CSV file for training:", type="csv")

    model = None
    if train_file is not None:
        train_data = pd.read_csv(train_file)
        st.write("Training Data:", train_data.head())

        # Train the model
        model, train_r2, train_rmse = train_model(train_data)

        if model is not None:
            # Display Training Metrics
            st.subheader("Training Data Metrics")
            st.write(f"R-Squared: {train_r2:.4f}")
            st.write(f"RMSE: {train_rmse:.4f}")

    # Test data upload
    st.header("Upload Test Data")
    test_file = st.file_uploader("Upload a CSV file for testing:", type="csv")

    if test_file is not None and model is not None:
        test_data = pd.read_csv(test_file)
        if 'cnt' not in test_data.columns:
            st.error("The test dataset must include the dependent variable 'cnt'.")
        else:
            y_test = test_data['cnt']
            X_test = preprocess_data(test_data)
            if X_test is not None:
                try:
                    y_test_pred = model.predict(X_test)
                    test_rmse = calculate_rmse(y_test, y_test_pred)

                    # Display Test Metrics
                    st.subheader("Test Data Metrics")
                    st.write(f"Test RMSE: {test_rmse:.4f}")
                except ValueError as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.error("Test data preprocessing failed.")

if __name__ == "__main__":
    main()

