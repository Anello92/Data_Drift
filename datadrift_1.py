# Monitoring, Identification, and Mitigation of Model and Data Drift
# Script 1: Initial Model Training

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
file = 'dataset.csv'
wine_data = pd.read_csv(file, delimiter=';')

# Separate features and target
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the initial model
model_v1 = RandomForestClassifier(random_state=42)
model_v1.fit(X_train_scaled, y_train)

# Evaluate the model on the initial test set
predictions = model_v1.predict(X_test_scaled)
accuracy = round(accuracy_score(y_test, predictions), 2) * 100

print("\nInitial Model Training")
print("\nInitial Model Accuracy (%):", accuracy)

# Save the trained model and scaler
model_file = 'model_v1.pkl'
scaler_file = 'scaler_v1.pkl'
joblib.dump(model_v1, model_file)
joblib.dump(scaler, scaler_file)

print(f"\nModel saved: {model_file}")
print(f"\nScaler saved: {scaler_file}")

print("\nScript 1 Executed Successfully!\n")
