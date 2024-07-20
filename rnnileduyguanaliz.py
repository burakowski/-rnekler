from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
import numpy as np
vocab_size = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size) 
print(x_train[0])


word_idx = imdb.get_word_index()
word_idx = {i: word for word, i in word_idx.items()}
print([word_idx[i] for i in x_train[0]])

print("Max length of a review:: ", len(max((x_train+x_test), key=len)))
print("Min length of a review:: ", len(min((x_train+x_test), key=len)))


from tensorflow.keras.preprocessing import sequence 
# Tüm yorumların sabit uzunluğunu maksimum 400 kelimeye kadar tut
max_words = 400
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)
x_valid, y_valid = x_train[:64], y_train[:64]
x_train_, y_train_ = x_train[64:], y_train[64:]


# her kelimenin yerleştirme boyutunu 32 olacak şekilde sabitle
embd_len = 32
 # RNN modeli oluştur
RNN_model = Sequential(name="Simple_RNN")
RNN_model.add(Embedding(vocab_size,                        embd_len,                        input_length=max_words)) 
# Yığılmış (birden fazla RNN katmanı) olması durumunda
# return_sequences=True'u kullanın
RNN_model.add(SimpleRNN(128,                        activation='tanh',                        return_sequences=False))
RNN_model.add(Dense(1, activation='sigmoid')) 
# model özetini yazdır
print(RNN_model.summary()) 
# modeli derle
RNN_model.compile(    loss="binary_crossentropy",    optimizer='adam',    metrics=['accuracy'])




# modelin eğitimi
history = RNN_model.fit(x_train_, y_train_,                        batch_size=64,                        epochs=5,                        verbose=1,                        validation_data=(x_valid, y_valid))
# sonuçların yazdırılması
print()
print("Simple_RNN Score---> ", RNN_model.evaluate(x_test, y_test, verbose=0))




 # GRU modelinin tanımlanması
gru_model = Sequential(name="GRU_Model")
gru_model.add(Embedding(vocab_size,                        embd_len,                        input_length=max_words))
gru_model.add(GRU(128,                  activation='tanh',                  return_sequences=False))
gru_model.add(Dense(1, activation='sigmoid'))
# model özetinin yazdırılması
print(gru_model.summary()) 
# modelin derlenmesi
gru_model.compile(    loss="binary_crossentropy",    optimizer='adam',    metrics=['accuracy']) 
# modelin eğitimi
history2 = gru_model.fit(x_train_, y_train_,                         batch_size=64,                         epochs=5,                         verbose=1,                         validation_data=(x_valid, y_valid)) 
# test sonuçlarının yazdırılması
print()
print("GRU model Score---> ", gru_model.evaluate(x_test, y_test, verbose=0))


#lstm ile deneme 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

lstm_model = Sequential(name="LSTM_Model")
lstm_model.add(Embedding(vocab_size, embd_len, input_length=max_words))
lstm_model.add(LSTM(128, activation='tanh', return_sequences=False))
lstm_model.add(Dense(1, activation='sigmoid'))

print(lstm_model.summary())

lstm_model.compile(loss="binary_crossentropy",    optimizer='adam',    metrics=['accuracy'])




history2 = lstm_model.fit(x_train_,
                         y_train_,
                         batch_size=64,
                         epochs=5,
                         verbose=1,
                         validation_data=(x_valid, y_valid)) 
print()
print("LSTM model Score---> ", lstm_model.evaluate(x_test, y_test, verbose=0))






