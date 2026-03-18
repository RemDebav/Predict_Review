# 🎬 Predict Review - Sentiment Classifier (Naive Bayes)

Ce projet est un classificateur de sentiments basé sur le jeu de données **IMDB**. Il utilise un algorithme **Naive Bayes** implémenté en Python pour l'entraînement et une interface web interactive (HTML/JS) pour tester les prédictions en temps réel.

## 🚀 Fonctionnalités
* **Entraînement personnalisé** : Script Python pour traiter le dataset IMDB et générer un modèle.
* **Modèle optimisé** : Utilisation d'une fonction de lissage (`alpha`) pour amplifier les prédictions.
* **Interface Web** : Une page `index.html` moderne pour tester vos propres critiques de films.
* **Visualisation** : Graphique d'analyse de la précision en fonction des paramètres du modèle.

## 📂 Structure du projet
* `train.py` : Script pour entraîner le modèle sur le dataset IMDB.
* `test.py` : Script d'évaluation pour tester la précision du modèle.
* `model.pkl` : Le modèle entraîné sauvegardé (dictionnaires de mots et probabilités).
* `index.html` : L'interface utilisateur pour classer les critiques.
* `accuracy_vs_alpha.png` : Graphique montrant l'impact du paramètre alpha sur la précision.

## 🛠️ Installation et Utilisation

### 1. Prérequis
Assurez-vous d'avoir Python installé ainsi que la bibliothèque `pandas` :
```bash
pip install pandas
