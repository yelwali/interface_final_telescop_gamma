from flask import Flask, request, render_template, send_file, url_for
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import PowerTransformer, RobustScaler
import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Définition du modèle de réseau de neurones
class Net(nn.Module):
    def __init__(self, hiddenSize1, hiddenSize2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, hiddenSize1)
        self.fc2 = nn.Linear(hiddenSize1, hiddenSize2)
        self.fc3 = nn.Linear(hiddenSize2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Charger le modèle de réseau de neurones
neural_model = torch.load("model.pt", map_location=torch.device("cpu"), weights_only=False)
neural_model.eval()

# Charger les transformateurs
pt = joblib.load("power_transformer.pkl")
scaler = joblib.load("robust_scaler.pkl")

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Erreur lors du rendu de la page d'accueil: {str(e)}")
        # Renvoyer une réponse simple si le template n'est pas disponible
        return "Application de classification Gamma/Hadron. Erreur de chargement du template."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_choice = request.form['model_choice']

        if model_choice == 'logistic_regression':
            model = joblib.load("logistic_regression_model.pkl")
        elif model_choice == 'xgboost':
            model = pickle.load(open('xgboost_model.pkl', 'rb'))
        elif model_choice == 'adaboost':
            model = pickle.load(open('adaboost_model.pkl', 'rb'))
        elif model_choice == 'decision tree classifier':
            model = pickle.load(open('decision_tree_model.pkl', 'rb'))
        elif model_choice == 'Random forest classifier':
            model = joblib.load('random_forest_model.joblib')
        elif model_choice == 'svm':
            model = pickle.load(open('svm_model.pkl', 'rb'))

        # Récupération des caractéristiques
        feature_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
                         "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
        float_features = [float(request.form[feature].replace(',', '.')) for feature in feature_names]
        final_features = np.array([float_features])
        final_features = pt.transform(final_features)
        final_features = scaler.transform(final_features)

        if model_choice == 'Neural':
            input_tensor = torch.tensor(final_features, dtype=torch.float32)
            with torch.no_grad():
                output = neural_model(input_tensor)
                predicted_class = torch.argmax(output).item()
                result = "Gamma" if predicted_class == 0 else "Hadron"
        else:
            prediction = model.predict(final_features)
            output = round(prediction[0])
            result = "Gamma" if output == 0 else "Hadron"

        return render_template('index.html', prediction_text=f"Classification: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Erreur: {str(e)}")

@app.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        model_choice = request.form['model_choice']
        file = request.files['file']
        if not file:
            return render_template('index.html', prediction_text="Erreur : Aucun fichier fourni.")

        df = pd.read_csv(file)
        expected_columns = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
                            "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
        if not all(col in df.columns for col in expected_columns):
            return render_template('index.html', prediction_text="Erreur : Colonnes manquantes dans le fichier.")

        X = pt.transform(df[expected_columns])
        X = scaler.transform(X)

        if model_choice == 'Neural':
            input_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                output = neural_model(input_tensor)
                predictions = torch.argmax(output, axis=1).numpy()
        else:
            if model_choice == 'logistic_regression':
                model = joblib.load("logistic_regression_model.pkl")
            elif model_choice == 'xgboost':
                model = pickle.load(open('xgboost_model.pkl', 'rb'))
            elif model_choice == 'adaboost':
                model = pickle.load(open('adaboost_model.pkl', 'rb'))
            elif model_choice == 'decision tree classifier':
                model = pickle.load(open('decision_tree_model.pkl', 'rb'))
            elif model_choice == 'Random forest classifier':
                model = joblib.load('random_forest_model.joblib')
            elif model_choice == 'svm':
                model = pickle.load(open('svm_model.pkl', 'rb'))
            predictions = model.predict(X)

        df['Prediction'] = ['Gamma' if p == 0 else 'Hadron' for p in predictions]

        # Créer un fichier CSV pour téléchargement
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"prediction_result_{timestamp}.csv"
        df.to_csv(output_filename, index=False)

        return render_template('index.html',
                               prediction_text="Prédiction terminée. Vous pouvez télécharger les résultats ci-dessous.",
                               download_link=output_filename)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Erreur: {str(e)}")

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    # Vérification de l'existence des fichiers nécessaires
    print("Vérification des modèles et fichiers...")
    if not os.path.exists("model.pt"):
        print("AVERTISSEMENT: Fichier model.pt non trouvé!")
    if not os.path.exists("power_transformer.pkl"):
        print("AVERTISSEMENT: Fichier power_transformer.pkl non trouvé!")
    if not os.path.exists("robust_scaler.pkl"):
        print("AVERTISSEMENT: Fichier robust_scaler.pkl non trouvé!")
    if not os.path.exists("templates"):
        print("AVERTISSEMENT: Dossier templates non trouvé!")
    elif not os.path.exists("templates/index.html"):
        print("AVERTISSEMENT: Template index.html non trouvé!")
            
    print("Démarrage de l'application...")
    app.run(debug=True, host='0.0.0.0')