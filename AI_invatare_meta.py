import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class DateExperimentale:
    def __init__(self, cale_fisier):
        """
        Inițializează clasa pentru gestionarea datelor experimentale.
        :param cale_fisier: Calea către fișierul Excel.
        """
        self.date = pd.read_excel(cale_fisier)
        self.tfidf_vectorizere = {}  # Dicționar pentru TF-IDF vectorizatoare
        self.label_encoders = {}  # Dicționar pentru LabelEncoders
        self.sarcini = []

    def preprocesare_date(self):
        """
        Preprocesează datele eliminând spațiile, vectorizând textul și codificând categoriile.
        """
        # Eliminăm spațiile și uniformizăm denumirile coloanelor
        self.date.columns = self.date.columns.str.strip()
        self.date.columns = self.date.columns.str.lower().str.replace(" ", "_")

        # Selectăm coloanele relevante
        coloane_relevante = [
            "provenineta_membrana", "solutie_electromigrare_pol_electrod_anod",
            "solutie_electromigrare_pol_electrod_cathode", "tub_transfer_crown_ether",
            "concentratie_crown_ether", "solutie_organica_+eter_coroana",
            "izotopi_(ug/l)", "setari_aparat", "potential_electromigrare",
            "anod_25_ore", "catod_25_ore", "anod_48_ore", "catod_48_ore"
        ]
        self.date = self.date[coloane_relevante]

        # Înlocuim valorile lipsă cu media coloanei (pentru coloanele numerice)
        self.date = self.date.fillna(self.date.mean(numeric_only=True))

        # Vectorizăm textul complex
        categorice_text = self.date.select_dtypes(include=['object']).columns
        for col in categorice_text:
            vectorizator = TfidfVectorizer(max_features=10)
            vectori = vectorizator.fit_transform(self.date[col].astype(str))
            vectori_df = pd.DataFrame(
                vectori.toarray(),
                columns=[f"{col}_{i}" for i in range(vectori.shape[1])]
            )
            self.date = pd.concat([self.date.reset_index(drop=True), vectori_df], axis=1)
            self.date.drop(columns=[col], inplace=True)
            self.tfidf_vectorizere[col] = vectorizator

    def genereaza_sarcini(self):
        """
        Generează sarcini experimentale din date.
        """
        # Separăm intrările (X) de ieșiri (y)
        X = self.date.drop(columns=["anod_25_ore", "catod_25_ore", "anod_48_ore", "catod_48_ore"]).values
        y = self.date[["anod_25_ore", "catod_25_ore", "anod_48_ore", "catod_48_ore"]].values.astype(float)
        self.sarcini.append((X, y))

    def obtine_date_noi(self, selectie):
        """
        Pregătește datele pentru o sarcină nouă bazată pe selecția utilizatorului.
        :param selectie: Dicționar cu selecțiile utilizatorului.
        :return: Datele de intrare pentru o sarcină nouă.
        """
        vectori_noi = []
        for col, valoare in selectie.items():
            if col in self.tfidf_vectorizere:  # Dacă este text vectorizat, aplicăm TF-IDF
                vectorizator = self.tfidf_vectorizere[col]
                valoare_str = str(valoare)
                vector = vectorizator.transform([valoare_str]).toarray().flatten()
                vectori_noi.extend(vector)
            else:  # Pentru valori numerice
                vectori_noi.append(float(valoare))
        return np.array([vectori_noi], dtype=float)


class MetaInvatare:
    def __init__(self, alpha=0.01, beta=0.001):
        """
        Inițializează algoritmul de meta-învățare.
        :param alpha: Rata de învățare locală.
        :param beta: Rata de învățare globală.
        """
        self.alpha = alpha
        self.beta = beta
        self.theta = None

    def initializare_model(self, input_dim, output_dim=4):
        """
        Inițializează parametrii modelului.
        :param input_dim: Dimensiunea intrărilor.
        :param output_dim: Numărul de ieșiri.
        """
        self.theta = np.random.randn(input_dim, output_dim)

    def pierdere(self, X, y, theta):
        """
        Calculează funcția de pierdere (MSE).
        """
        pred = X @ theta
        return np.mean((pred - y) ** 2)

    def gradient(self, X, y, theta):
        """
        Calculează gradientul funcției de pierdere.
        """
        pred = X @ theta
        error = pred - y
        return 2 * X.T @ error / len(X)

    def adaptare_locala(self, X, y):
        """
        Ajustează parametrii pe baza unei sarcini specifice.
        """
        theta_local = self.theta.copy()
        for _ in range(10):
            grad = self.gradient(X, y, theta_local)
            theta_local -= self.alpha * grad
        return theta_local

    def actualizare_globala(self, sarcini):
        """
        Actualizează parametrii globali pe baza rezultatelor sarcinilor.
        """
        suma_gradiente = np.zeros_like(self.theta)
        for X, y in sarcini:
            theta_local = self.adaptare_locala(X, y)
            suma_gradiente += self.gradient(X, y, theta_local)
        self.theta -= self.beta * suma_gradiente / len(sarcini)

    def antrenare(self, sarcini, num_iteratii=3000):
        """
        Antrenează modelul utilizând meta-învățarea.
        """
        for iteratie in range(num_iteratii):
            self.actualizare_globala(sarcini)
            pierdere_medie = np.mean(
                [self.pierdere(X, y, self.theta) for X, y in sarcini]
            )
            print(f"Iterația {iteratie + 1}/{num_iteratii} - Pierdere medie: {pierdere_medie:.4f}")

    def prezicere(self, X):
        """
        Prezice valorile pentru un set de intrare.
        """
        return X @ self.theta
