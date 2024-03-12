import math
from graphviz import Graph
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform

# Charge le dataset
dataset = pd.read_csv('./dataset.csv')

# Les features catégoriques sont classées par proximité
# avec un indice représentant cette proximité
musics = {
    0: "other",
    0.1: "classical",
    0.2: "jazz",
    3: "hiphop",
    3.1: "trap",
    3.2: "rap",
    4.2: "rock",
    4.3: "metal",
    4.4: "technical death metal"
}

citys = {
    0: "lille",
    0.25: "paris",
    0.5: "toulouse",
    0.75: "marseille",
    6: "madrid",
}

jobs = {
    0: "doctor",
    2: "teacher",
    5: "fireman",
    7: "painter",
    10: "designer",
    11: "developper",
    12: "engineer"
}

# Define numerical and categorical features
numerical_features = ['age', 'height']
categorical_features = ['job', 'city', 'favorite music style']

# Define column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Fit and transform the data
X_processed = preprocessor.fit_transform(dataset)

def dissimilarity_cat(cat_x1, cat_x2, data, indexColomn):
    key1 = -1
    key2 = -1
    res = 0
    for key, value in data.items():
        if cat_x1.iloc[indexColomn] in value:
            key1 = key
        if cat_x2.iloc[indexColomn] in value:
            key2 = key
    if (key1 > key2):
        res = key1 - key2
    else:
        res = key2 - key1
    if res >= 3:
        return 6
    else:
        return res
    

def custom_dissimilarity(x1, x2):
    # Extraire les features 1 et 2 qui sont des features numérique
    num_x1 = x1[1:2]
    num_x2 = x2[1:2]
    
    # Extraire les features 3, 4 et 5 qui sont des features catégorique
    cat_x1 = x1[3:].astype(str)
    cat_x2 = x2[3:].astype(str)
    
    # Calculer la distance euclidienne pour les features numérique
    num_distance = np.linalg.norm(num_x1 - num_x2)
    # Limiter la distance à 6 pour éviter des valeurs trop importantes
    if (num_distance > 6):
        num_distance = 6
    
    # Calculer la distance des features catégorique
    music_distance = dissimilarity_cat(cat_x1, cat_x2, musics, 2)
    city_distance = dissimilarity_cat(cat_x1, cat_x2, citys, 1)
    job_distance = dissimilarity_cat(cat_x1, cat_x2, jobs, 0)

    # Définition des poids
    num_weight = 5
    music_weight = 4
    city_weight = 5
    job_weight = 1
    
    # 2 villes sont distantes de 2 si elles sont dans le meme pays
    # 2 musiques sont distantes de 2 si elles sont dans le meme genre musical
    if (0 < city_distance) & (city_distance < 1):
        city_distance = 2
    if (0 < music_distance) & (music_distance < 1):
        music_distance = 2
    
    # Combiner les distances avec les poids
    dissimilarity = math.sqrt(
        (num_weight * num_distance)
        + music_distance * music_weight
        + city_distance * city_weight
        + job_distance * job_weight
    )
    print("\n\n--------compare-----------")
    print(x1)
    print("------------with------------ ")
    print(x2)
    print("----------distences--------- ")
    print("num: ")
    print(num_distance)
    print("music: ")
    print(music_distance)
    print("city: ")
    print(city_distance)
    print("job: ")
    print(job_distance)
    print("--------dissimilarity------- ")
    print(dissimilarity)
    return dissimilarity

# Creation de la matrice de dissimilarité
num_samples = len(dataset.index)
dissimilarity_matrix = np.zeros((num_samples, num_samples))

for i in range(num_samples):
    for j in range(num_samples):
        dissimilarity_matrix[i, j] = custom_dissimilarity(dataset.iloc[i], dataset.iloc[j])

# Calculer la moyenne et l'écart-type de la matrice de dissimilarité
mean_dissimilarity = np.mean(dissimilarity_matrix)
std_dissimilarity = np.std(dissimilarity_matrix)

# Enregistrer la matrice dans un fichier npy
np.save('dissimilarity_matrix.npy', dissimilarity_matrix)

# Afficher la moyenne et l'écart-type
print("Mean Dissimilarity:", mean_dissimilarity)
print("Standard Deviation of Dissimilarity:", std_dissimilarity)
print ("Loading file ...")
loaded_dissimilarity_matrix = np.load('dissimilarity_matrix.npy')
print(loaded_dissimilarity_matrix)

#--------graphique--------------

# Limite pour ajouter un lien de dissimilarité entre 2 features
threshold = 8.7

dot = Graph(comment="Graph created from complex data", strict=True)
for index in range(num_samples):
    # iloc[5] est la colonne où est stocker la catégorie "favorite music style"
    fav_music = dataset.loc[index].iloc[5]
    dot.node(fav_music)

# On compare tous les samples entre eux
for sample_1_id in range(num_samples):
    for sample_2_id in range(num_samples):
        # On evite de comparer un sample avec lui meme
        if not sample_1_id == sample_2_id:
            player_1_name = dataset.loc[sample_1_id].iloc[5]
            player_2_name = dataset.loc[sample_2_id].iloc[5]
            if dissimilarity_matrix[sample_1_id, sample_2_id] > threshold:
                # Un trait représentant la dissimilarité est tracer sur le graphique
                dot.edge(
                    player_1_name,
                    player_2_name,
                    color="darkolivegreen4",
                    penwidth="1.1",
                )

# On affiche le graphique
dot.attr(label=f"threshold {threshold}", fontsize="20")
graph_name = f"images/complex_data_threshold_{threshold}"
dot.render(graph_name)
