import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Paramètres pour la régression logistique
param_grid_LR = {
    'solver': ['lbfgs'],             # Algorithme utilisé pour l'optimisation
    'penalty': ['l2'],               # Type de pénalité (L2 utilise Ridge)
    'C': [0.001],                    # Force de régularisation (plus C est petit, plus la régularisation est forte)
    'intercept_scaling': [0.001]     # Redimensionnement de l'ordonnée à l'origine (constante de régularisation)
}

# Paramètres pour random forest
param_grid_RF = {
    'n_estimators': [50],           # Nombre d'arbres dans la forêt
    'max_depth': [3],               # Profondeur maximale des arbres (évite l'overfitting)
    'min_samples_split': [5],       # Nombre minimum d'échantillons requis pour diviser un noeud interne
    'min_samples_leaf': [12],       # Nombre minimum d'échantillons requis pour être une feuille (dernier noeud en bas de l'arbre)
    'max_features': ['sqrt']        # Nombre de caractéristiques à considérer lors de la recherche du meilleur split
}

# Initialize les Cross Validators 
cv_LR = StratifiedKFold(n_splits=7, shuffle=True, random_state=41)
cv_RF = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def modelPredict(model):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Regression Linéaire
grid_search_LR = GridSearchCV(LogisticRegression(), param_grid=param_grid_LR, cv=cv_LR, scoring='accuracy')
grid_search_LR.fit(X_train, y_train)
best_model_LR = grid_search_LR.best_estimator_
print("\n________Linear Regression________\n")
print(f"Test accuracy: {modelPredict(best_model_LR)}")
print(f"Train accuracy: {accuracy_score(y_train, best_model_LR.predict(X_train))}")
print(f"Best hyperparameters: {best_model_LR.get_params()}")

# Random Forest
grid_search_RF = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_RF, cv=cv_RF, scoring='accuracy')
grid_search_RF.fit(X_train, y_train)
best_model_RF = grid_search_RF.best_estimator_
print("\n___________Random Forest_________\n")
print(f"Test accuracy: {modelPredict(best_model_RF)}")
print(f"Train accuracy: {accuracy_score(y_train, best_model_RF.predict(X_train))}")
print(f"Best hyperparameters: {best_model_RF.get_params()}")


#--------------Graphique-----------------

feature_importances = best_model_RF.feature_importances_
# Random Forest calcul l'importance des differentes features sur le modèle,
# on affiche un graph de ces features classées par importance

feature_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), sorted_feature_importances, align='center')
plt.xticks(range(len(feature_importances)), sorted_feature_names, rotation=90)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()