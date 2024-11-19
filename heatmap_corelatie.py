
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Încărcare date
file_path = 'tabel_completat.xlsx'
df = pd.ExcelFile(file_path).parse('Sheet1')

# Preprocesare date
features = df[['Provenineta Membrana', 
               'Solutie electromigrare pol electrod Anod', 
               'Solutie electromigrare pol electrod Cathode', 
               'Tub Transfer Crown Ether', 
               'Concentratie Crown Ether', 
               'Potential electromigrare']]

# Transformare date categorice
for column in features.select_dtypes(include='object').columns:
    encoder = LabelEncoder()
    features[column] = encoder.fit_transform(features[column])

# Normalizare
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Calcul matrice de corelație
correlation_matrix = pd.DataFrame(features_scaled, columns=[
    'Provenineta Membrana',
    'Solutie Anod',
    'Solutie Cathode',
    'Tub Transfer',
    'Concentratie Crown Ether',
    'Potential electromigrare'
]).corr()

# Vizualizare matrice de corelație ca heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Matricea de Corelație a Caracteristicilor')
plt.xlabel('Caracteristici')
plt.ylabel('Caracteristici')
plt.show()
