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
st.markdown('This application utilizes a Random Forest model to predict rainfall in Australia, leveraging 10 years of observational data')
st.image('images/australia.jpg')

# Заголовок секції з характеристиками рослини
st.header("Weather parameters")

location = st.text_input('Location')
RainToday = st.checkbox('Rain Today')

MinTemp = st.slider('Minimal Temperature', -100, 100, 0)
Maxtemp = st.slider('Maximum Temperature', -100, 100, 0)
Rainfall = st.slider('Rainfall', -100, 100, 0)
Evaporation = st.slider('Evaporation', -100, 100, 0)
Sunshine = st.slider('Sunshine', -100, 100, 0)
WindGustDir = st.slider('Wind Gust Direction', -100, 100, 0)
WindGustSpeed = st.slider('Wind Gust Speed', -100, 100, 0)

col1, col2 = st.columns(2)

with col1:
   
    WindDir9am = st.slider('Wind Direction at 9am', -100, 100, 0)
    WindSpeed9am = st.slider('Wind Speed at 9am', -100, 100, 0)
    Humidity9am = st.slider('Humidity at 9am', -100, 100, 0)
    Pressure9am = st.slider('Pressure at 9am', -100, 100, 0)
    Cloud9am = st.slider('Cloud Cover at 9am', -100, 100, 0)
    Temp9am = st.slider('Temperature at 9am', -100, 100, 0)
      
with col2:
    WindDir3pm = st.slider('Wind Direction at 3pm', -100, 100, 0)
    WindSpeed3pm = st.slider('Wind Speed at 3pm', -100, 100, 0)
    Humidity3pm = st.slider('Humidity at 3pm', -100, 100, 0)
    Pressure3pm = st.slider('Pressure at 3pm', -100, 100, 0)
    Cloud3pm = st.slider('Cloud Cover at 3pm', -100, 100, 0)
    Temp3pm = st.slider('Temperature at 3pm', -100, 100, 0)


# Кнопка для прогнозування
if st.button("Predict the rain tomorrow"):
    # Викликаємо функцію прогнозування
    result = predict()
    st.write(f"Result: {result}")