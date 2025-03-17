import pandas as pd
import sys

path = "C:/Users/anapa/Documents/UC/S4/DL/Git/TallerRentHouse/RentHouse/src/"
sys.path.append(path)

from data_loader import load_data
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
import numpy as np

dataset_path = r"C:\Users\anapa\Documents\UC\S4\DL\Git\TallerRentHouse\House_Rent_Dataset.csv"
df = load_data(dataset_path)

X = df[['BHK', 'Size', 'Bathroom']]
y = df['Rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(3,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  
    ])
    
model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'mse', metrics = ['mae'])