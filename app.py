import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('house_price_model.pkl')

# Title of the app
st.title("Ames Housing Price Prediction")

# Collect user input for features
square_footage = st.number_input("Enter the square footage of the house (Gr Liv Area)", min_value=0)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025)
overall_quality = st.selectbox("Overall Quality (1-10)", options=[i for i in range(1, 11)])
overall_condition = st.selectbox("Overall Condition (1-10)", options=[i for i in range(1, 11)])
year_remod_add = st.number_input("Year Remodeled/Add", min_value=1800, max_value=2025)
exterior_qual = st.selectbox("Exterior Quality (1-5)", options=[i for i in range(1, 6)])
total_bsmnt_sf = st.number_input("Total Basement Square Footage", min_value=0)
garage_area = st.number_input("Garage Area (sq ft)", min_value=0)

# Collect other feature inputs here as needed, following the same format

# Create a DataFrame from user inputs
user_input = pd.DataFrame({
    'Gr Liv Area': [square_footage],
    'Year Built': [year_built],
    'Overall Qual': [overall_quality],
    'Overall Cond': [overall_condition],
    'Year Remod/Add': [year_remod_add],
    'Exter Qual': [exterior_qual],
    'Total Bsmt SF': [total_bsmnt_sf],
    'Garage Area': [garage_area],
    # Add any other necessary features here
})

# Predict the house price
predicted_price = model.predict(user_input)

# Display the predicted price in a larger, more prominent format
st.subheader("Predicted House Price:")
st.markdown(f"<h1 style='text-align: center; color: green;'>${predicted_price[0]:,.2f}</h1>", unsafe_allow_html=True)



