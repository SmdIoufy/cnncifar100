# Classification d'images avec un CNN ( CIFAR-100)

TP de Deep Learning (Master 1 Data Science et Génie Logiciel).

## Objectif
Construire un réseau de neurones convolutif (CNN) pour classer des images :
-
- **CIFAR-100** : 100 classes (exercice)

## Données
- 60 000 images couleur de 32 × 32 pixels (50 000 d'entraînement, 10 000 de test).
- Chargées avec `cifar10.load_data()` et `cifar100.load_data(label_mode='fine')` (Keras).
- Normalisation : valeurs des pixels divisées par 255.

## Architecture
3 blocs `Conv2D` (3×3, ReLU, padding same) + `BatchNormalization` + `MaxPooling2D (2×2)`,
puis `Flatten`, deux couches `Dense` avec `Dropout`, et une sortie `softmax`.

## Méthode
- Recherche d'hyperparamètres avec **Keras Tuner (Hyperband)** : nombre de filtres, neurones des couches Dense, taux de dropout.
- Entraînement avec **EarlyStopping** (`patience=3`) sur `val_loss`.
- Optimiseur `adam`, perte `sparse_categorical_crossentropy`.

## Résultats
| Jeu de données | Accuracy test | Accuracy train |
|---|---|---|
| CIFAR-10 | 73,1 % | 80,3 % |
| CIFAR-100 | 44,3 % | 62,1 % |

Sur CIFAR-100, l'écart train/test indique un début de sur-apprentissage.

.

## Lancer le projet
1. Ouvrir le notebook dans Google Colab (activer le GPU).
2. Exécuter les cellules dans l'ordre.

## Outils
Python, TensorFlow/Keras, Keras Tuner, NumPy, Matplotlib
