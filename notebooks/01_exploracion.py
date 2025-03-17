import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

path = "C:/Users/anapa/Documents/UC/S4/DL/Git/TallerRentHouse/RentHouse/src/"
sys.path.append(path)

from data_loader import load_data

dataset_path = r"C:\Users\anapa\Documents\UC\S4\DL\Git\TallerRentHouse\House_Rent_Dataset.csv"
X_train, X_test, y_train, y_test = load_data(dataset_path)

df_train = pd.DataFrame(X_train)
df_train['Rent'] = y_train  # Agregar la variable objetivo

df_test = pd.DataFrame(X_test)
df_test['Rent'] = y_test  # Agregar la variable objetivo

print("Información del dataset:")
print(df_train.info())

print("\nDescripción estadística:")
print(df_train.describe())

# Gráfico de distribución de la variable objetivo
plt.figure(figsize=(8, 5))
sns.histplot(df_train['Rent'], kde=True, bins=30)
plt.title("Distribución de la variable objetivo")
plt.xlabel("Precio de alquiler")
plt.ylabel("Frecuencia")
plt.show()

# Gráfico de correlaciones
plt.figure(figsize=(10, 6))
sns.heatmap(df_train.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlacion")
plt.show()

print(df_train.corr().columns)
print(df_train.corr().columns)
