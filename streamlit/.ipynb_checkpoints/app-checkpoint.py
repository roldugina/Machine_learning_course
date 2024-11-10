import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict():
    model = joblib.load('models/aussie_rain.joblib')
    #data = np.expand_dims(np.array([sepal_l, sepal_w, petal_l, petal_w]), axis=0)
    #predictions = model.predict(data)
    #return predictions[0]
    return 0

# App title
st.title('Weather Forecasting')
st.markdown('This is an app for forecasting weather in Australia based on Random Forest model')
st.image('images/australia.jpg')

# Заголовок секції з характеристиками рослини
st.header("Weather parameters")
col1, col2 = st.columns(2)

with col1:
    location = st.text_input('Location')
    MinTemp = st.number_input('Minimal Temperature')
    Maxtemp = st.number_input('Maximum Temperature')
    Rainfall = st.checkbox('Rainfall')
    
with col2:
    Temp9am = st.number_input('Temperature at 9 am')
    Temp3pm = st.number_input('Temperature at 3 pm')

# Кнопка для прогнозування
if st.button("Forecast the weather"):
    # Викликаємо функцію прогнозування
    result = predict()
    st.write(f"Result: {result}")