import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor

# Load the dataset
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Preprocess the data if necessary (e.g., scaling)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Split the training data into training and validation sets
x_train_split, x_val, y_train_split, y_val = train_test_split(x_train_scaled, y_train, test_size=0.2, random_state=42)

# Reshape the target variables to 1D arrays
y_train_split = np.ravel(y_train_split)
y_val = np.ravel(y_val)

# Initialize models
models = {
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'MLP Regressor': MLPRegressor(),
    'SVR': SVR(),
    'AdaBoost Regressor': AdaBoostRegressor()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(x_train_split, y_train_split)
    y_pred = model.predict(x_val)
    r2 = r2_score(y_val, y_pred)
    results[name] = r2

# Choose the best model based on validation set performance
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

# Evaluate the best model on the test set
y_pred_test = best_model.predict(x_test_scaled)
r2_test = r2_score(y_test, y_pred_test)

print(f"Best model: {best_model_name}")
print(f"Validation R2 score: {results[best_model_name]}")
print(f"Test R2 score: {r2_test}")

# TODO : varier les hyper parametre pour trouver le meilleur