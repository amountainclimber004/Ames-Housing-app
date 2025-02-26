import streamlit as st
import pandas as pd
import joblib
import os

st.title('Ames Housing Price Predictor')

# Check if the model file exists
model_path = 'housing_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found. Please upload the model.")
    st.stop()

# Load model
model = joblib.load(model_path)

# Load reference dataset for dropdowns (Ensure 'data.csv' exists)
data_path = 'data.csv'
if not os.path.exists(data_path):
    st.error("Error: Reference dataset 'data.csv' not found. Please provide a dataset for dropdown selections.")
    st.stop()

df = pd.read_csv(data_path)

# Input widgets
gr_liv_area = st.slider('Living Area (sqft)', 500, 5000, 1500)
total_bsmt_sf = st.slider('Basement Area (sqft)', 0, 3000, 1000)
bedrooms = st.selectbox('Bedrooms', [1, 2, 3, 4, 5])
neighborhood = st.selectbox('Neighborhood', df['Neighborhood'].unique() if 'Neighborhood' in df else [])
house_style = st.selectbox('House Style', df['HouseStyle'].unique() if 'HouseStyle' in df else [])

# Create input dataframe
input_data = pd.DataFrame([[gr_liv_area, total_bsmt_sf, bedrooms, neighborhood, house_style]],
                          columns=['GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr', 'Neighborhood', 'HouseStyle'])

# Ensure categorical encoding matches the trained model
# Example: If label encoding was used in training, apply the same here
# You may need to load an encoder and transform categorical variables

# Prediction
if st.button('Predict Price'):
    try:
        prediction = model.predict(input_data)
        st.success(f'Predicted House Price: ${prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"Error in prediction: {e}")

