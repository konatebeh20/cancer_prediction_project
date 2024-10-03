import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import pandas as pd
from model import load_and_preprocess_data, create_and_train_model, evaluate_model
import os

def browse_file():
    filename = filedialog.askopenfilename(title="Sélectionner un fichier CSV", filetypes=[("CSV files", "*.csv")])
    if filename:
        entry_file.delete(0, tk.END)
        entry_file.insert(0, filename)

def start_prediction():
    file_path = entry_file.get()
    if not file_path:
        messagebox.showerror("Erreur", "Veuillez sélectionner un fichier de données.")
        return
    
    try:
        # Charger et préparer les données
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
        
        # Entraîner le modèle
        trace, model = create_and_train_model(X_train, y_train)
        
        # Évaluer le modèle
        auc_roc = evaluate_model(X_test, y_test, trace, model)
        
        # Afficher le résultat
        label_result.config(text=f"AUC-ROC: {auc_roc:.2f}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")

# Interface graphique
root = tk.Tk()
root.title("Prédiction du cancer")

# Exemple de contenu de l'interface
label = ttk.Label(root, text="Bienvenue dans l'application de prédiction du cancer")
label.pack(padx=20, pady=20)

# Sélection de fichier
frame_file = tk.Frame(root)
frame_file.pack(pady=10)
label_file = tk.Label(frame_file, text="Fichier de données (CSV):")
label_file.pack(side=tk.LEFT, padx=5)
entry_file = tk.Entry(frame_file, width=40)
entry_file.pack(side=tk.LEFT, padx=5)
button_browse = tk.Button(frame_file, text="Parcourir", command=browse_file)
button_browse.pack(side=tk.LEFT)

# Bouton de prédiction
button_predict = tk.Button(root, text="Commencer la prédiction", command=start_prediction)
button_predict.pack(pady=10)

# Affichage du résultat
label_result = tk.Label(root, text="Résultat: AUC-ROC sera affiché ici")
label_result.pack(pady=10)

# Lancer la boucle principale de l'interface Tkinter
root.mainloop()
