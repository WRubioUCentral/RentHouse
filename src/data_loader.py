import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):

    df = pd.read_csv(file_path) ## Lee path
    ## print("Columnas disponibles en el dataset:", df.columns.tolist())
    ## print("Columnas independientes:", df[['BHK', 'Size', 'Bathroom']].head(3))
    ## print("Columna objetivo:", df['Rent'].head(3))
    numericas = df.select_dtypes(include=['number']) ## Columnas numericas
    independientes = df[['BHK', 'Size', 'Bathroom']] ## Columnas independientes
    objetivo = df['Rent'] ## Columna objetivo
    
    df = df.select_dtypes(include=['number'])
    
    y = objetivo.values
    X = independientes.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = r"C:\Users\anapa\Documents\UC\S4\DL\Git\TallerRentHouse\House_Rent_Dataset.csv"
    X_train, X_test, y_train, y_test = load_data(file_path)
    print("\nDatos cargados y preprocesados correctamente.")