# Delivery Time Prediction

This project aims to predict the expected time it would take for a delivery driver to find a parking spot and reach the delivery spot using a TensorFlow-based machine learning model. The model is trained on geographical, temporal, traffic, and environmental data to make accurate predictions.

## Project Structure

- `delivery_data.csv`: Sample data used for training the model.
- `train_model.py`: Script to train the machine learning model.
- `app.py`: Flask application to serve the model and accept REST GET requests for predictions.
- `delivery_time_prediction_model.h5`: Trained TensorFlow model.

## Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- `pip` package manager

Install the necessary Python packages:

```sh
pip install pandas numpy scikit-learn tensorflow flask
```

## Training the Model

To train the model, use the `train_model.py` script. Ensure you have your training data in `delivery_data.csv` or modify the script to point to your data source.

```sh
python train_model.py
```

This script will preprocess the data, train the model, and save the trained model to `delivery_time_prediction_model.h5`.

## Running the Flask Server

The Flask server loads the trained model and sets up an endpoint to accept GET requests for predictions.

Start the Flask server by running:

```sh
python app.py
```

The server will start and listen on `http://0.0.0.0:5000`.

## Using the API

### Predict Delivery Time

Endpoint: `/predict`

Method: `GET`

#### Query Parameters

- `latitude`: Latitude of the delivery location (float).
- `longitude`: Longitude of the delivery location (float).
- `time_of_day`: Hour of the day (0-23) (int).
- `day_of_week`: Day of the week (0 for Sunday, 1 for Monday, ..., 6 for Saturday) (int).
- `traffic_level`: Traffic congestion level (e.g., 1 to 5) (int).
- `weather_condition`: Weather condition (e.g., clear, rain, snow, cloudy) (string).
- `type_of_location`: Type of delivery location (e.g., residential, commercial, industrial) (string).

#### Example Request

```sh
curl "http://localhost:5000/predict?latitude=40.712776&longitude=-74.005974&time_of_day=14&day_of_week=2&traffic_level=3&weather_condition=clear&type_of_location=residential"
```

#### Example Response

```json
{
  "predicted_delivery_time": 15.0
}
```

## Sample Data

A sample of how your `delivery_data.csv` file should look:

```csv
latitude,longitude,time_of_day,day_of_week,traffic_level,weather_condition,type_of_location,delivery_time
40.712776,-74.005974,14,2,3,clear,residential,15
34.052235,-118.243683,10,5,2,rain,commercial,20
51.507351,-0.127758,18,6,5,snow,industrial,25
48.856613,2.352222,8,1,1,clear,residential,10
35.689487,139.691711,12,3,4,cloudy,commercial,18
```

## Notes

- Ensure your data is clean and properly formatted before training the model.
- The preprocessing steps should match between training and serving the model to ensure consistency.