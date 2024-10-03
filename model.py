import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pymc3 as pm
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data(file_path):
    # Charger les données
    data = pd.read_csv(file_path)
    X = data.drop(['id', 'diagnosis'], axis=1)
    y = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def create_and_train_model(X_train, y_train):
    # Créer et entraîner le modèle bayésien
    def create_model(X, y):
        with pm.Model() as model:
            beta = pm.Normal('beta', mu=0, sd=1, shape=X.shape[1])
            alpha = pm.Normal('alpha', mu=0, sd=10)
            p = pm.math.invlogit(alpha + pm.math.dot(X, beta))
            y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
        return model
    
    with create_model(X_train, y_train) as model:
        trace = pm.sample(2000, tune=1000, cores=2)
    return trace, model

def evaluate_model(X_test, y_test, trace, model):
    # Prédictions et évaluation
    ppc = pm.sample_posterior_predictive(trace, samples=1000, model=model)
    y_pred = ppc['y_obs'].mean(axis=0)
    
    # Calcul de l'AUC-ROC
    auc_roc = roc_auc_score(y_test, y_pred)
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC-ROC: {auc_roc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend()
    
    # Sauvegarder l'image
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig('images/roc_curve.png')
    
    # Rapport de classification
    y_pred_class = (y_pred > 0.5).astype(int)
    report = classification_report(y_test, y_pred_class)
    print(report)
    
    # Sauvegarder le rapport dans un fichier texte
    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    
    return auc_roc

# Utilisation de la fonction pour charger et prétraiter les données
file_path = 'chemin/vers/votre/fichier.csv'  # Remplacez par le chemin vers votre fichier
X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

# Affichage des dimensions des données
print("Dimensions des données:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Créer et entraîner le modèle
trace, model = create_and_train_model(X_train, y_train)

# Évaluer le modèle
auc_roc = evaluate_model(X_test, y_test, trace, model)
print(f"AUC-ROC: {auc_roc:.2f}")











# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pymc3 as pm
# from sklearn.metrics import roc_auc_score, roc_curve, classification_report
# import matplotlib.pyplot as plt
# import os

# def load_and_preprocess_data(file_path):
#     # Charger les données
#     data = pd.read_csv(file_path)
#     X = data.drop(['id', 'diagnosis'], axis=1)
#     y = data['diagnosis'].map({'M': 1, 'B': 0})
    
#     # Division en ensembles d'entraînement et de test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Normalisation des données
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     return X_train_scaled, X_test_scaled, y_train, y_test

# print("Dimensions des données:")
# print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
# print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# def create_and_train_model(X_train, y_train):
#     # Créer et entraîner le modèle bayésien
#     def create_model(X, y):
#         with pm.Model() as model:
#             beta = pm.Normal('beta', mu=0, sd=1, shape=X.shape[1])
#             alpha = pm.Normal('alpha', mu=0, sd=10)
#             p = pm.math.invlogit(alpha + pm.math.dot(X, beta))
#             y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
#         return model
    
#     with create_model(X_train, y_train) as model:
#         trace = pm.sample(2000, tune=1000, cores=2)
#     return trace, model

# def evaluate_model(X_test, y_test, trace, model):
#     # Prédictions et évaluation
#     ppc = pm.sample_posterior_predictive(trace, samples=1000, model=model)
#     y_pred = ppc['y_obs'].mean(axis=0)
    
#     # Calcul de l'AUC-ROC
#     auc_roc = roc_auc_score(y_test, y_pred)
    
#     # Courbe ROC
#     fpr, tpr, _ = roc_curve(y_test, y_pred)
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, label=f'AUC-ROC: {auc_roc:.2f}')
#     plt.plot([0, 1], [0, 1], linestyle='--')
#     plt.xlabel('Taux de faux positifs')
#     plt.ylabel('Taux de vrais positifs')
#     plt.title('Courbe ROC')
#     plt.legend()
    
#     # Sauvegarder l'image
#     if not os.path.exists('images'):
#         os.makedirs('images')
#     plt.savefig('images/roc_curve.png')
    
#     # Rapport de classification
#     y_pred_class = (y_pred > 0.5).astype(int)
#     report = classification_report(y_test, y_pred_class)
#     print(report)
    
#     # Sauvegarder le rapport dans un fichier texte
#     if not os.path.exists('results'):
#         os.makedirs('results')
#     with open('results/classification_report.txt', 'w') as f:
#         f.write(report)
    
#     return auc_roc
