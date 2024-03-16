# Importing necessary libraries
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor

# Load the dataset
X_train = np.load('./X_train.npy')
X_test = np.load('./X_test.npy')
y_train = np.load('./y_train.npy')
y_test = np.load('./y_test.npy')

# Preprocessing: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a function to train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

# Initialize models
models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "MLPRegressor": MLPRegressor(),
    "SVR": SVR(),
    "AdaBoostRegressor": AdaBoostRegressor()
}

# Flatten y_train and y_test
y_train_flat = y_train.ravel()
y_test_flat = y_test.ravel()

# Train and evaluate models
results = {}
for name, model in models.items():
    r2 = train_and_evaluate_model(model, X_train_scaled, y_train_flat, X_test_scaled, y_test_flat)
    results[name] = r2

# Print results
for name, r2 in results.items():
    print(f"{name}: R2 Score = {r2}")

# Select the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Test the best model on the test data
best_model.fit(X_train_scaled, y_train_flat)
y_pred_test = best_model.predict(X_test_scaled)
r2_test = r2_score(y_test_flat, y_pred_test)
print(f"Test R2 Score of the Best Model: {r2_test}")
