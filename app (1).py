! pip install streamlit pandas numpy joblib xgboost scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("best_xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Get the correct feature names that the scaler was trained on
training_feature_names = scaler.get_feature_names_out()  # Use this to get the correct feature names

# Define feature columns (ensure these match your dataset)
feature_columns = training_feature_names.tolist()  # Update feature_columns

# ... (rest of your Streamlit code)

# Get additional economic indicators (Replace with your data source)
# Assuming your data source has the same feature names as training_feature_names
default_data = {feature: np.random.uniform(0, 1) for feature in feature_columns}

# Convert input data into DataFrame
input_data = pd.DataFrame([default_data])

# ... (rest of the code)
# Define feature columns (ensure these match your dataset)
feature_columns = ['GDP Growth', 'Inflation Rate', 'Debt-to-GDP Ratio', 'Total Reserves (% of total external debt)', 
                   'Short-term Debt (% of total external debt)']

# 🎨 Streamlit App UI
st.title("🌍 Sovereign Debt Crisis Prediction App")
st.markdown("### Enter Country and Year to Predict Sovereign Debt Crisis")

# User Inputs
country = st.text_input("Enter Country Name", "Japan")
year = st.number_input("Enter Year", min_value=2000, max_value=2030, value=2025)

# Get additional economic indicators (In a real app, this data would come from a database)
default_data = {
    'GDP Growth': np.random.uniform(-5, 10),
    'Inflation Rate': np.random.uniform(0, 15),
    'Debt-to-GDP Ratio': np.random.uniform(20, 150),
    'Total Reserves (% of total external debt)': np.random.uniform(5, 80),
    'Short-term Debt (% of total external debt)': np.random.uniform(1, 40),
}

# Convert input data into DataFrame
input_data = pd.DataFrame([default_data])

# ... (Previous code)

# Get the correct feature names that the scaler was trained on
training_feature_names = scaler.get_feature_names_out()  # Use this to get the correct feature names

# Ensure input_data has the correct columns (and only those columns)
input_data = input_data.reindex(columns=training_feature_names, fill_value=0) # Fill missing columns with 0

# Scale input features
input_data_scaled = scaler.transform(input_data)

# ... (Rest of the code)

# Scale input features
input_data_scaled = scaler.transform(input_data)

# Predict Crisis (1 = Crisis, 0 = No Crisis)
if st.button("🔍 Predict Crisis"):
    prediction = model.predict(input_data_scaled)[0]
    result = "🔴 Crisis" if prediction == 1 else "🟢 No Crisis"
    
    st.subheader(f"Prediction for {country} in {year}:")
    st.markdown(f"## {result}")


