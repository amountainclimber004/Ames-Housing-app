import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Define the input fields for user input
st.title("Ames Housing Price Prediction App")

st.sidebar.header("Enter House Features:")
overall_qual = st.sidebar.slider("Overall Quality (Poor-High)", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
garage_cars = st.sidebar.slider("Number of Cars in the Garage", 0, 5, 2)
total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2024, value=2000)

# Make prediction when user clicks the button
if st.sidebar.button("Predict Price"):
    input_data = np.array([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, year_built]])
    predicted_price = model.predict(input_data)[0]

    st.subheader("Predicted Sale Price:")
    st.write(f"💰 {<h1 style='text-align: center; color: green;'>$predicted_price:,.2f}")

st.sidebar.markdown("---")
st.sidebar.text("Built for Streamlit")




