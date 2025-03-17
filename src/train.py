import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_data
from model import create_model

file_path = r"C:\Users\anapa\Documents\UC\S4\DL\Git\TallerRentHouse\House_Rent_Dataset.csv"
X_train, X_test, y_train, y_test = load_data(file_path)

input_shape = X_train.shape[1]
model = create_model(input_shape)

epochs = 100
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = epochs,
    batch_size = 32,
    verbose = 1
)

model.save(r"C:\Users\anapa\Documents\UC\S4\DL\Git\TallerRentHouse\RentHouse\models\rental_price_model.keras")
model.save(r"C:\Users\anapa\Documents\UC\S4\DL\Git\TallerRentHouse\RentHouse\models\rental_price_model.h5")
print("Modelo guardado con éxito como rental_price_model.keras y rental_price_model.h5")

plt.figure(figsize = (9,4))
plt.plot(history.history['loss'], label = 'Loss_train')
plt.plot(history.history['val_loss'], label = 'Loss_validation')
plt.xlabel('Epocas')
plt.ylabel('Perdida')
plt.title('Evolucion de la Perdida durante el Entrenamiento')
plt.legend()
plt.show()