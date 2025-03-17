import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore

def create_model(input_shape):

    model = keras.Sequential([
        layers.Input(shape = (input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    sample_input_shape = 3 
    model = create_model(sample_input_shape)
    model.summary()