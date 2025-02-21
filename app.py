import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('housing_model.pkl')

st.title('Ames Housing Price Predictor')

# Input widgets
gr_liv_area = st.slider('Living Area (sqft)', 500, 5000, 1500)
total_bsmt_sf = st.slider('Basement Area (sqft)', 0, 3000, 1000)
bedrooms = st.selectbox('Bedrooms', [1, 2, 3, 4, 5])
neighborhood = st.selectbox('Neighborhood', df['Neighborhood'].unique())
house_style = st.selectbox('House Style', df['HouseStyle'].unique())

# Create input dataframe
input_data = pd.DataFrame([[gr_liv_area, total_bsmt_sf, bedrooms, neighborhood, house_style]],
                          columns=['GrLivArea', 'TotalBsmtSF', 'BedroomAbvGr', 'Neighborhood', 'HouseStyle'])

# Prediction
if st.button('Predict Price'):
    prediction = model.predict(input_data)
    st.success(f'Predicted House Price: ${prediction[0]:,.2f}')
