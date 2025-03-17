import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import load_data


model = tf.keras.models.load_model(
   "models/rental_price_model.h5",
    compile = False
)
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

nuevos_datos = np.array([[1200, 3, 2]])  # (Size, BHK, Bathroom)

file_path = r"C:\Users\anapa\Documents\UC\S4\DL\Git\TallerRentHouse\House_Rent_Dataset.csv"
X_train, X_test, y_train, y_test = load_data(file_path)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)  # Estadisticas de entrenamiento
nuevos_datos = (nuevos_datos - mean) / std

print(f"Numero de características usadas en el modelo: {X_train.shape[1]}")

y_pred = model.predict(nuevos_datos)
print(f"Prediccion de precio de alquiler: {y_pred[0][0]:.2f}")