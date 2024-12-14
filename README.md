# Weather Forecasting with Deep Learning

## Description
This project focuses on developing a weather forecasting application using deep learning models. The model uses weather data collected from various provinces in Vietnam to predict the temperature for a city based on climate parameters such as humidity, rainfall, and wind speed. This application is implemented using Flask and allows users to input the name of a city to receive weather forecast information.

## Project Structure
1. **`app.py`**: The main file for the Flask application, where the machine learning model is loaded, and the weather forecasting functionality is implemented.
2. **`index.html`**: The user interface where users can input the name of a city and view the temperature forecast result.
3. **`processed_weather_data.csv`**: The processed weather data containing features like humidity, rainfall, wind speed, and temperature.
4. **`weather_prediction_model.h5`**: The deep learning model (LSTM) trained to predict temperature based on weather features.
5. **`Predict_Temperature.ipynb`**: Jupyter notebook containing code for training and evaluating the temperature prediction model.

