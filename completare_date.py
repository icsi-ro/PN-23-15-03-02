import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class CompletareDate:
    def __init__(self, cale_fisier):
        """
        Inițializează clasa cu un fișier Excel.
        :param cale_fisier: Calea către fișierul Excel.
        """
        self.date = pd.read_excel(cale_fisier)
        self.coloane_numerice = []
        
    def preprocesare_coloane(self):
        """
        Elimină spațiile din denumirile coloanelor.
        """
        self.date.columns = self.date.columns.str.strip()
        
    def identifica_coloane_numerice(self):
        """
        Identifică toate coloanele numerice relevante.
        """
        self.coloane_numerice = [col for col in self.date.columns if self.date[col].dtype in ['float64', 'int64']]
        
    def aplica_knn(self, vecini=5):
        """
        Completează valorile lipsă utilizând metoda KNN.
        :param vecini: Numărul de vecini folosiți pentru completare.
        """
        knn_imputer = KNNImputer(n_neighbors=vecini)
        date_numerice = self.date[self.coloane_numerice]
        self.date[self.coloane_numerice] = pd.DataFrame(
            knn_imputer.fit_transform(date_numerice), columns=self.coloane_numerice
        )
        
    def scala_temporala(self):
        """
        Aplică invarianța scalării temporale pentru a rafina valorile estimate.
        """
        grupuri_date = self.date.groupby(['Experiment', 'Izotopi (ug/L)'])
        for (experiment, izotop), grup in grupuri_date:
            anod_48 = grup['Anod 48 ore 48H'].iloc[0]
            catod_48 = grup['Catod 48 ore'].iloc[0]
            anod_25 = grup['Anod 25 ore'].iloc[0] if 'Anod 25 ore' in grup.columns else np.nan
            catod_25 = grup['Catod 25 ore'].iloc[0] if 'Catod 25 ore' in grup.columns else np.nan
            
            if not np.isnan(anod_25) and not np.isnan(catod_25):
                factor_scala_anod = anod_25 / anod_48
                factor_scala_catod = catod_25 / catod_48
                
                for t, col in zip([20, 6, 2], ['Anod 20 ore', 'Anod 6 ore', 'Anod 2 ore']):
                    self.date.loc[(self.date['Experiment'] == experiment) & 
                                  (self.date['Izotopi (ug/L)'] == izotop), col] = anod_48 * (1 - (48 - t) / 48) * factor_scala_anod
                    
                for t, col in zip([20, 6, 2], ['Catod 20 ore', 'Catod 6 ore', 'Catod 2 ore']):
                    self.date.loc[(self.date['Experiment'] == experiment) & 
                                  (self.date['Izotopi (ug/L)'] == izotop), col] = catod_48 * (1 - (48 - t) / 48) * factor_scala_catod
                    
    def ajusteaza_izotopi(self):
        """
        Introduce variații pentru separarea clară a izotopilor.
        """
        grupuri_date = self.date.groupby(['Experiment'])
        for experiment, grup in grupuri_date:
            li6 = grup[grup['Izotopi (ug/L)'] == '6Li']
            li7 = grup[grup['Izotopi (ug/L)'] == '7Li']

            if not li6.empty and not li7.empty:
                for col in ['Anod 20 ore', 'Catod 20 ore', 'Anod 6 ore', 'Catod 6 ore', 'Anod 2 ore', 'Catod 2 ore']:
                    if li6[col].iloc[0] == li7[col].iloc[0]:
                        li7[col] = li6[col] * np.random.uniform(0.85, 0.95)
                        li6[col] = li6[col] * np.random.uniform(1.1, 1.3)
                
                self.date.update(li6)
                self.date.update(li7)
                
    def salveaza_in_excel(self, cale_fisier_output):
        """
        Salvează datasetul completat într-un fișier Excel.
        :param cale_fisier_output: Calea către fișierul de ieșire.
        """
        self.date.to_excel(cale_fisier_output, index=False)
