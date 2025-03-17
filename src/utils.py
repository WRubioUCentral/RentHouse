import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def normalize_data(X, scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler.transform(X)

def denormalize_data(y, scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler.inverse_transform(y.reshape(-1, 1)).flatten()

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

if __name__ == "__main__":
    model = load_model(r"C:\Users\anapa\Documents\UC\S4\DL\Git\TallerRentHouse\RentHouse\models\rental_price_model.keras")
    print("Modelo cargado correctamente.")