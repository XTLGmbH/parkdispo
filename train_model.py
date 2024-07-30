import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('delivery_data.csv')

# Feature columns
features = [
    'latitude', 'longitude', 'time_of_day', 'day_of_week',
    'traffic_level', 'weather_condition', 'type_of_location'
]

# Target column
target = 'delivery_time'

# Preprocess the data
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

# Split the data
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing and modeling pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, epochs=50, batch_size=32, verbose=0))
])

# Define the model
def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=[X_train.shape[1]]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
loss, mae = model.score(X_test, y_test)
print(f'Mean Absolute Error: {mae}')

# Save the trained model
model.named_steps['classifier'].model.save('delivery_time_prediction_model.h5')
