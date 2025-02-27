import streamlit as st
import pandas as pd
import joblib

# Load the trained model (make sure the path to the model is correct)
model = joblib.load('house_price_model.pkl')

# Sample feature inputs (replace with actual inputs or input widgets)
square_footage = st.number_input("Enter Square Footage", min_value=100, max_value=10000, step=1)
num_bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=10, step=1)
num_bathrooms = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10, step=1)

# Create the input data (this is where you need to ensure proper format)
input_data = pd.DataFrame({
    'Gr Liv Area': [square_footage],
    'Bedroom AbvGr': [num_bedrooms],
    'Full Bath': [num_bathrooms]
})

# Make the prediction
predicted_price = model.predict(input_data)[0]

# Display the result
st.subheader("Predicted Sale Price:")
st.markdown(f"<h1 style='text-align: center; color: green;'>${predicted_price:,.2f}</h1>", unsafe_allow_html=True)





