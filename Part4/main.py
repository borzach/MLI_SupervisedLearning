import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
import optuna

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Preprocessing data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Split the training data into training and validation sets
x_train_split, x_val, y_train_split, y_val = train_test_split(x_train_scaled, y_train, test_size=0.2, random_state=42)

# Flattening the array
y_train_split = np.ravel(y_train_split)
y_val = np.ravel(y_val)

# Objectives function for each models
def objective_ridge(trial):
    alpha = trial.suggest_float('alpha', 1e-3, 10, log=True)
    alpha = trial.suggest_float('alpha', 1e-3, 10, log=True)
    model = Ridge(alpha=alpha)
    model.fit(x_train_split, y_train_split)
    y_pred = model.predict(x_val)
    return r2_score(y_val, y_pred)

def objective_lasso(trial):
    alpha = trial.suggest_float('alpha', 1e-3, 10, log=True)
    model = Lasso(alpha=alpha)
    model.fit(x_train_split, y_train_split)
    y_pred = model.predict(x_val)
    return r2_score(y_val, y_pred)

def objective_svr(trial):
    C = trial.suggest_float('C', 1e-3, 10, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-3, 1, log=True)
    model = SVR(C=C, epsilon=epsilon)
    model.fit(x_train_split, y_train_split)
    y_pred = model.predict(x_val)
    return r2_score(y_val, y_pred)

def objective_adaboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 1, log=True)
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(x_train_split, y_train_split)
    y_pred = model.predict(x_val)
    return r2_score(y_val, y_pred)

# Optimizing models
study_ridge = optuna.create_study(direction='maximize', study_name="Ridge study")
study_ridge.optimize(objective_ridge, n_trials=100)

study_lasso = optuna.create_study(direction='maximize', study_name="Lasso study")
study_lasso.optimize(objective_lasso, n_trials=100)

study_svr = optuna.create_study(direction='maximize', study_name="SVR study")
study_svr.optimize(objective_svr, n_trials=100)

study_adaboost = optuna.create_study(direction='maximize', study_name="AdaBoost study")
study_adaboost.optimize(objective_adaboost, n_trials=100)

# Storing best hyperparameters
best_params_ridge = study_ridge.best_params
best_params_lasso = study_lasso.best_params
best_params_svr = study_svr.best_params
best_params_adaboost = study_adaboost.best_params

# Evaluating models with their new hyperparam in order to obtain the new r2_score of each
best_ridge = Ridge(**best_params_ridge)
best_ridge.fit(x_train_scaled, y_train)
y_pred_test_ridge = best_ridge.predict(x_test_scaled)
r2_test_ridge = r2_score(y_test, y_pred_test_ridge)

best_lasso = Lasso(**best_params_lasso)
best_lasso.fit(x_train_scaled, y_train)
y_pred_test_lasso = best_lasso.predict(x_test_scaled)
r2_test_lasso = r2_score(y_test, y_pred_test_lasso)

best_svr = SVR(**best_params_svr)
best_svr.fit(x_train_scaled, y_train)
y_pred_test_svr = best_svr.predict(x_test_scaled)
r2_test_svr = r2_score(y_test, y_pred_test_svr)

best_adaboost = AdaBoostRegressor(**best_params_adaboost)
best_adaboost.fit(x_train_scaled, y_train)
y_pred_test_adaboost = best_adaboost.predict(x_test_scaled)
r2_test_adaboost = r2_score(y_test, y_pred_test_adaboost)

# Results
print("Ridge Regression:")
print(f"Best hyperparams: {best_params_ridge}")
print(f"R2 score: {r2_test_ridge}\n")

print("Lasso Regression:")
print(f"Best hyperparams: {best_params_lasso}")
print(f"R2 score: {r2_test_lasso}\n")

print("SVR:")
print(f"Best hyperparams: {best_params_svr}")
print(f"R2 score: {r2_test_svr}\n")

print("AdaBoost Regressor:")
print(f"Best hyperparams: {best_params_adaboost}")
print(f"R2 score: {r2_test_adaboost}\n")
