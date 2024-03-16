import csv

def valeurs_uniques_csv(chemin_fichier, colonne):
    valeurs_uniques = set()  # Utilisation d'un ensemble pour stocker les valeurs uniques
    
    with open(chemin_fichier, newline='') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv)
        
        # Ignorer l'en-tête si nécessaire
        next(lecteur_csv)
        
        for ligne in lecteur_csv:
            if len(ligne) > colonne:  # Vérifier si la ligne a suffisamment de colonnes
                valeur = ligne[colonne]
                valeurs_uniques.add(valeur)
    
    return valeurs_uniques

# Chemin vers votre fichier CSV
chemin_fichier_csv = './dataset.csv'

# Numéro de colonne (indexé à partir de 0)
colonne_a_analyser = 3  # Pour la deuxième colonne, utiliser l'index 1

# Récupération des valeurs uniques
valeurs_uniques = valeurs_uniques_csv(chemin_fichier_csv, colonne_a_analyser)

# Affichage des valeurs uniques
print("Valeurs uniques de la colonne 2:")
for valeur in valeurs_uniques:
    print(valeur)
