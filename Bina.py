from keras.models import Model
from keras.layers import Dense, Dropout, Input
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.callbacks import EarlyStopping

# Veriyi yükleme
dataset = pd.read_excel("bina.xlsx")
dataset = dataset.values
X = dataset[:, 0:8]
y = dataset[:, 8:10]

# Veriyi eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Veriyi ölçekleme
sc = StandardScaler()
data_x_train_scaled = sc.fit_transform(X_train)
data_x_test_scaled = sc.transform(X_test)

# NumPy array'e dönüştürme
data_x_train_scaled, data_x_test_scaled, data_y_train, data_y_test = \
    np.array(data_x_train_scaled), np.array(data_x_test_scaled), np.array(y_train), np.array(y_test)

# Model tanımlama
input_layer = Input(shape=(data_x_train_scaled.shape[1],), name="Input_Layer")
common_path = Dense(units=128, activation='relu', name='First_Dense')(input_layer)
common_path = Dropout(0.3)(common_path)
common_path = Dense(units=128, activation='relu', name='Second_Dense')(common_path)
common_path = Dropout(0.3)(common_path)

first_output = Dense(units=1, name='first_Output__Last_Layer')(common_path)
second_output_path = Dense(units=64, activation='relu', name='Second_Output__First_Dense')(common_path)
second_output_path = Dropout(0.3)(second_output_path)
second_output = Dense(units=1, name='Second_Output__Last_layer')(second_output_path)

model = Model(inputs=input_layer, outputs=[first_output, second_output])
print(model.summary())

# Modeli derleme
optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)
model.compile(optimizer=optimizer,
              loss={'first_Output__Last_Layer': 'mse', 'Second_Output__Last_layer': 'mse'},
              metrics={'first_Output__Last_Layer': tf.keras.metrics.RootMeanSquaredError(),
                       'Second_Output__Last_layer': tf.keras.metrics.RootMeanSquaredError()})

# EarlyStopping tanımlama
EarlyStopping_callback = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       verbose=1)

# Modeli eğitme
history = model.fit(x=data_x_train_scaled, y=[data_y_train[:, 0], data_y_train[:, 1]], verbose=0, epochs=500,
                    batch_size=10, validation_split=0.3, callbacks=[EarlyStopping_callback])

# Tahmin yapma
y_pred = np.array(model.predict(data_x_test_scaled))

# R2 skorlarını hesaplama
from sklearn.metrics import r2_score
print("ilk çıkışın R2 değeri:", r2_score(data_y_test[:, 0], y_pred[0, :].flatten()))
print("ikinci çıkışın R2 değeri:", r2_score(data_y_test[:, 1], y_pred[1, :].flatten()))

# İlk çıkış için RMSE kayıp değerleri grafiği
plt.plot(history.history['first_Output__Last_Layer_root_mean_squared_error'])
plt.plot(history.history['val_first_Output__Last_Layer_root_mean_squared_error'])
plt.title('Model\'in ilk çıkış için RMSE kayıp değerleri')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# İkinci çıkış için RMSE kayıp değerleri grafiği
plt.figure()
plt.plot(history.history['Second_Output__Last_layer_root_mean_squared_error'])
plt.plot(history.history['val_Second_Output__Last_layer_root_mean_squared_error'])
plt.title('Model\'in ikinci çıkış için RMSE kayıp değerleri')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
