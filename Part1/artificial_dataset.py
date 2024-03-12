import csv
import numpy as np

np.random.seed(42)

# Générer les colonnes et les lignes
num_rows = 300
num_cols = 6

# Générer la moyenne et l'écart type pour chaque colonne
means = np.random.uniform(low=0, high=10, size=num_cols)
std_devs = np.random.uniform(low=0.1, high=3, size=num_cols)

# Générer une matrice de corrélation
correlation_matrix = np.random.uniform(low=-1, high=1, size=(num_cols, num_cols))
np.fill_diagonal(correlation_matrix, 1)  # Définir la diagonale à 1
correlation_matrix = np.dot(correlation_matrix, correlation_matrix.T)

# Utiliser la matrice de corrélation pour générer des données
data = np.random.multivariate_normal(means, np.diag(std_devs), size=num_rows)
data = np.dot(data, np.linalg.cholesky(correlation_matrix).T)

# Convertir une colonne de float a int
int_column_index = np.random.randint(0, num_cols)
data[:, int_column_index] = data[:, int_column_index].astype(int)

# Enregistrer les données dans un fichier CSV
fichier_csv = "artificial_dataset.csv"
with open(fichier_csv, mode='w', newline='') as fichier:
    ecrivain = csv.writer(fichier)
    ecrivain.writerow([f"Colonne_{i}" for i in range(1, num_cols+1)])
    ecrivain.writerows(data)

print(f"Le fichier CSV '{fichier_csv}' a été généré avec succès.")