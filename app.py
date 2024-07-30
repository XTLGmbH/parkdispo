import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Define the feature columns and preprocessors
numeric_features = ['latitude', 'longitude', 'time_of_day', 'traffic_level']
categorical_features = ['day_of_week', 'weather_condition', 'type_of_location']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Load the trained model
model = tf.keras.models.load_model('delivery_time_prediction_model.h5')

# Define a function to preprocess the input data
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    preprocessed_data = preprocessor.transform(df)
    return preprocessed_data

# Define a route to handle predictions
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get parameters from the request
        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
        time_of_day = int(request.args.get('time_of_day'))
        day_of_week = int(request.args.get('day_of_week'))
        traffic_level = int(request.args.get('traffic_level'))
        weather_condition = request.args.get('weather_condition')
        type_of_location = request.args.get('type_of_location')

        # Create a dictionary for the input data
        input_data = {
            'latitude': latitude,
            'longitude': longitude,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'traffic_level': traffic_level,
            'weather_condition': weather_condition,
            'type_of_location': type_of_location
        }

        # Preprocess the input data
        preprocessed_data = preprocess_input(input_data)

        # Make a prediction
        prediction = model.predict(preprocessed_data)
        predicted_time = prediction[0][0]

        # Return the prediction as a JSON response
        return jsonify({'predicted_delivery_time': predicted_time})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Fit the preprocessor to some dummy data to initialize it
    dummy_data = pd.DataFrame({
        'latitude': [0],
        'longitude': [0],
        'time_of_day': [0],
        'day_of_week': [0],
        'traffic_level': [0],
        'weather_condition': ['clear'],
        'type_of_location': ['residential']
    })
    preprocessor.fit(dummy_data)

    app.run(host='0.0.0.0', port=5000)
