import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

path = "./house-votes-84.csv"
datasets = [pd.read_csv(path)]

figure = plt.figure(figsize=(27, 9))
i = 1

for ds_cnt, ds in enumerate(datasets):
    dataset_cleaned = ds.replace('?', np.NaN, inplace=False)
    imputer = SimpleImputer(strategy='mean')
    dataset_cleaned_imputed = pd.DataFrame(imputer.fit_transform(dataset_cleaned.drop(columns=['party'])), columns=dataset_cleaned.columns[1:])
    le = LabelEncoder()
    dataset_cleaned_imputed['party'] = le.fit_transform(dataset_cleaned['party'])
    X = dataset_cleaned_imputed.drop(columns=['party'])
    y = dataset_cleaned_imputed['party']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    feature_names = X.columns

    x_min, x_max = X.min(axis=0) - 0.5, X.max(axis=0) + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    if ds_cnt == 0: 
        ax.set_title("Input data")
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")

    ax.set_xlim(X_train_pca[:, 0].min(), X_train_pca[:, 0].max())
    ax.set_ylim(X_train_pca[:, 1].min(), X_train_pca[:, 1].max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
    
        clf.fit(X_train_pca, y_train)
        score = clf.score(X_test_pca, y_test)
    
        DecisionBoundaryDisplay.from_estimator(
            clf, X_train_pca, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )
    
        ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
    
        ax.set_xlim(X_train_pca[:, 0].min(), X_train_pca[:, 0].max())
        ax.set_ylim(X_train_pca[:, 1].min(), X_train_pca[:, 1].max())
    
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            X_train_pca[:, 0].max() - 0.3,
            X_train_pca[:, 1].min() + 0.3,
            ("%.2f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        i += 1
    

plt.tight_layout()
plt.show()
