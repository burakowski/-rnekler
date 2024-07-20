import numpy as np 
import pandas as pd
from tensorflow.keras.datasets import mnist
import seaborn as sbn
import matplotlib.pyplot as plt

(trainX,trainY),(testX,testY) = mnist.load_data()

print('Training data shapes :X=%s,Y=%s' % (trainX.shape,trainY.shape))
print('Training data shapes :X=%s,Y=%s' % (testX.shape,testY.shape))

for j in range(5):
    i = np.random.randint(0,10000)
    plt.subplot(550+1+j)
    plt.imshow(trainX [i],cmap='gray')
    plt.title(trainY[i])
plt.show()

#iki boyutlu veriyi tek boyuta indirgeme


trainX = trainX/255
testX = testX/255
train_data = np.reshape(trainX,(60000,28*28))
test_data = np.reshape(testX,(10000,28*28))
print(train_data.shape,test_data.shape)

#outncoder

import tensorflow
input_data = tensorflow.keras.layers.Input(shape= (784,))
encoder = tensorflow.keras.layers.Dense(100)(input_data)  #encoder = tf.keras.layers.Dense(100, activation='relu')(input_data) kısa yazımı
encoder = tensorflow.keras.layers.Activation('relu')(encoder)
encoder = tensorflow.keras.layers.Dense(50)(input_data)
encoder = tensorflow.keras.layers.Activation('relu')(encoder)
encoder = tensorflow.keras.layers.Dense(25)(input_data)
encoder = tensorflow.keras.layers.Activation('relu')(encoder)
encoded = tensorflow.keras.layers.Dense(2)(encoder)


decoder = tensorflow.keras.layers.Dense(25)(encoded)
decoder = tensorflow.keras.layers.Activation('relu')(decoder)
decoder = tensorflow.keras.layers.Dense(50)(decoder)
decoder = tensorflow.keras.layers.Activation('relu')(decoder)
decoder = tensorflow.keras.layers.Dense(100)(decoder)
decoder = tensorflow.keras.layers.Activation('relu')(decoder)
decoder = tensorflow.keras.layers.Dense(784)(decoder)

autoencoder = tensorflow.keras.models.Model(inputs=input_data, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')


autoencoder.summary()


# Modeli eğitme
autoencoder.fit(train_data, train_data, epochs=10, batch_size=64, validation_data=(test_data, test_data))


#gerçek resimler 
for i in range(5):
    plt.subplot(350+1+i)
    plt.imshow(testX[i],cmap= 'gray')
plt.show()



for i in range(5):
    plt.subplot(550+1+i)
    output = autoencoder.predict(np.array([test_data[i]]))
    op_image = np.reshape(output[0]*255,(28,28))
    plt.imshow(op_image,cmap='gray')
plt.show()    
    