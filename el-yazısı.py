import numpy as np 
import matplotlib.pyplot as plt 
from keras.layers import Dense, Flatten
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist

# Veriyi yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)

# Kategorik veriye çevir
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Modeli oluştur
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.summary()

# Modeli derle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Modeli eğit
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Tahmin yap
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

# Grafikleri çiz
fig, axes = plt.subplots(ncols=10, sharex=False, sharey=True, figsize=(20,4))

for i in range(10):
    axes[i].set_title(predictions[i])
    axes[i].imshow(X_test[i], cmap='gray')
    axes[i].get_xaxis().set_visible(False)  # Düzeltilen satır
    axes[i].get_yaxis().set_visible(False)

plt.show()