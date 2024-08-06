# Monitoring, Identification, and Mitigation of Model and Data Drift
# Mitigation Strategy 3 - Retrain the Model with New Data, Optimize Hyperparameters, Combine Different Algorithms

# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
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

# Define the seed for reproducibility
np.random.seed(541)

# Function to simulate Data Drift
def simulate_data_drift(X, drift_factor):
    drifted_X = X + np.random.normal(0, drift_factor, X.shape)
    return drifted_X

# Simulate Data Drift
X_test_drifted = simulate_data_drift(X_test_scaled, drift_factor=0.7)

# Mitigation Strategy: Retrain the model with updated data and hyperparameter optimization
def retrain_model(X_train, y_train, X_new, y_new):
    
    X_combined = np.vstack((X_train, X_new))
    y_combined = np.hstack((y_train, y_new))

    # Define the hyperparameters for optimization
    param_grid_gbc = {
        'n_estimators': [400, 500, 600],
        'learning_rate': [0.0001, 0.001, 0.01],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    param_grid_svc = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Create new models
    gbc = GradientBoostingClassifier(random_state=42)
    svc = SVC(probability=True, random_state=42)
    
    # Apply GridSearchCV to find the best hyperparameters
    grid_search_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid_gbc, cv=3, n_jobs=-1, verbose=2)
    grid_search_gbc.fit(X_combined, y_combined)
    
    grid_search_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=3, n_jobs=-1, verbose=2)
    grid_search_svc.fit(X_combined, y_combined)

    # Train the models with the best hyperparameters
    best_gbc = grid_search_gbc.best_estimator_
    best_svc = grid_search_svc.best_estimator_

    # Create an ensemble of the best models
    ensemble_model = VotingClassifier(estimators=[('gbc', best_gbc), ('svc', best_svc)], voting='soft')
    ensemble_model.fit(X_combined, y_combined)
    
    return ensemble_model

# Simulate new training data (for simplicity, reuse X_test_drifted)
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_test_drifted, y_test, test_size=0.3, random_state=42)

print("\nHyperparameter Optimization...\n")

# Retrain the model with new data and hyperparameter optimization
model_v2 = retrain_model(X_train_scaled, y_train, X_new_train, y_new_train)

# Function to monitor model performance
def monitor_model_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Evaluate the retrained model
accuracy_post_retrain = round(monitor_model_performance(model_v2, X_new_test, y_new_test), 2)

print("\nScript 4 - Data Drift Mitigation Strategy")

print("\nAccuracy After Model Retraining with New Data and Hyperparameter Optimization:", accuracy_post_retrain)

# Save the retrained model and scaler
model_file = 'model_v4.pkl'
scaler_file = 'scaler_v4.pkl'
joblib.dump(model_v2, model_file)
joblib.dump(scaler, scaler_file)

print(f"\nModel saved: {model_file}")
print(f"\nScaler saved: {scaler_file}")

print("\nScript 4 Executed Successfully!\n")