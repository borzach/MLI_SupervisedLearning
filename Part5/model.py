import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Source of the dataset : https://www.kaggle.com/datasets/balakrishcodes/others/data
file_path = './house-votes-84.csv'
ds = pd.read_csv(file_path)

# Change missing value by null
ds.replace('?', np.NaN, inplace=True)

# Split the X (value) and Y (target), here party is our target because we want to predict the vote
X = ds.drop(columns=['party'])
y = ds['party']

# Replace missing value using mean along each column
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Encode the 'party' column (change from string to binary value (0 = democrat ,1 = republican))
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Slice the dataset between training and testing, we use 70% of the dataset for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.3, random_state=42)

# Initializing the KNeighborsClassifier with our best params
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train) # Training the model
y_pred = knn.predict(X_test) # Predicting targets value

# Plotting and getting metrics  
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy_pred = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f'Model Accuracy: {accuracy_pred * 100:.2f}%')
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {roc_auc:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Ploting confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, ax=ax[0])
ax[0].set_title('Confusion Matrix for KNeighborsClassifier')

# Ploting ROC Curve
ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic (ROC) Curve')
ax[1].legend(loc="lower right")

plt.tight_layout()
plt.show()