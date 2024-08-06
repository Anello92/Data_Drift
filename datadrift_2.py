# Monitoring, Identification, and Mitigation of Model and Data Drift
# Script 2: Data Drift Simulation and Model Evaluation

# Imports
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load the dataset
file = 'dataset.csv'
wine_data = pd.read_csv(file, delimiter=';')

# Separate features and target
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load the trained model and scaler
model_file = 'model_v1.pkl'
scaler_file = 'scaler_v1.pkl'
model_v1 = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# Standardize the test data
X_test_scaled = scaler.transform(X_test)

# Define the seed for reproducibility
np.random.seed(41)

# Function to simulate Data Drift
def simulate_data_drift(X, drift_factor):
    drifted_X = X + np.random.normal(0, drift_factor, X.shape)
    return drifted_X

# Simulate Data Drift
X_test_drifted = simulate_data_drift(X_test_scaled, drift_factor=0.7)

# Evaluate the model on the test set with Data Drift
drifted_predictions = model_v1.predict(X_test_drifted)
drifted_accuracy = accuracy_score(y_test, drifted_predictions)

# Function to monitor model performance
def monitor_model_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Monitor model performance before and after drift
pre_drift_accuracy = round(monitor_model_performance(model_v1, X_test_scaled, y_test), 2)
post_drift_accuracy = round(monitor_model_performance(model_v1, X_test_drifted, y_test), 2)

print("\nScript 2 - Data Drift Simulation and Model Evaluation")

print("\nAccuracy Before Data Drift:", pre_drift_accuracy)

print("\nAccuracy After Data Drift:", post_drift_accuracy)

print("\nScript 2 Executed Successfully!\n")