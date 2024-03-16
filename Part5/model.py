import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Source of the dataset : https://www.kaggle.com/datasets/balakrishcodes/others/data
path = "./house-votes-84.csv"
dataset = pd.read_csv(path)

print("First few rows of the dataset:")
print(dataset.head())

print("\nBasic statistics:")
print(dataset.describe())

# Because there is ? value as string in the dataset, we replace them by null so we can handle them
dataset_cleaned = dataset.replace('?', np.NaN, inplace=False)

# Check for missing values
print("\nMissing values:")
print(dataset_cleaned.isnull().sum())

# Visualize distribution of votes on different issues
plt.figure(figsize=(12, 6))
dataset_cleaned.drop(columns=['party']).apply(pd.Series.value_counts).plot(kind='bar', stacked=True)
plt.title('Distribution of votes on different issues')
plt.xlabel('Vote')
plt.ylabel('Count')
plt.legend(title='Party')
plt.xticks(rotation=45)
#plt.show()

# We use a function to compute the missing values
# imputer = SimpleImputer(strategy='most_frequent') | Here we try the most_frequent algorithm but the accuracy was slightly less better
imputer = SimpleImputer(strategy='mean')
dataset_cleaned_imputed = pd.DataFrame(imputer.fit_transform(dataset_cleaned.drop(columns=['party'])), columns=dataset_cleaned.columns[1:])

# Encode categorical variables
le = LabelEncoder()
dataset_cleaned_imputed['party'] = le.fit_transform(dataset_cleaned['party'])


# Split data into features and target
X = dataset_cleaned_imputed.drop(columns=['party'])
y = dataset_cleaned_imputed['party']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data - feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
dt_model = KNeighborsClassifier(3)

# Evaluate logistic regression model
dt_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_cv_score = cross_val_score(dt_model, X_train_scaled, y_train, cv=5).mean()

print("\nDecision tree model:")
print("Accuracy:", dt_accuracy)
print("Cross-validation Score:", dt_cv_score)
print("\nClassification Report:")
print(classification_report(y_test, dt_pred))
