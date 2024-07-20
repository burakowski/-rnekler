import ccxt
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Binance borsasından veriyi çekme
ex = ccxt.binance()
df = pd.DataFrame(
    ex.fetch_ohlcv(symbol='BTC/USDT', timeframe='1d', limit=1000),
    columns=['unix', 'open', 'high', 'low', 'close', 'volume']
)
df['date'] = pd.to_datetime(df.unix, unit='ms')

# Grafik ayarları
plt.rcParams['figure.figsize'] = [14, 8]  # [width, height]
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# Kapanış fiyatlarının grafiğini çizme
ax = df.plot(x='date', y='close')
ax.set_xlabel('Date')
ax.set_ylabel('Price ($)')
ax.set_title('BTC Daily Close Price')  # Burada set_title kullanıldı

plt.show()

# Veriyi ölçekleme
scaler = MinMaxScaler()
close_price = df.close.values.reshape(-1,1)
scaled_close = scaler.fit_transform(close_price)

# Fonksiyonlar
seq_len = 60
def split_into_sequences(data, seq_len):    
    n_seq = len(data) - seq_len + 1    
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(data, seq_len, train_frac):    
    sequences = split_into_sequences(data, seq_len)    
    n_train = int(sequences.shape[0] * train_frac)    
    x_train = sequences[:n_train, :-1, :]    
    y_train = sequences[:n_train, -1, :]    
    x_test = sequences[n_train:, :-1, :]    
    y_test = sequences[n_train:, -1, :]    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_train_test_sets(scaled_close, seq_len, train_frac=0.9)

# Veriler hazırlandıktan sonra LSTM modelini oluşturmaya başlayabiliriz.
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential

dropout = 0.2
window_size = seq_len - 1  # Burada normal tire (-) kullanıldı

model = Sequential()
model.add(LSTM(window_size, return_sequences=True, input_shape=(window_size, x_train.shape[-1])))
model.add(Dropout(rate=dropout))
model.add(Bidirectional(LSTM(window_size * 2, return_sequences=True))) 
model.add(Dropout(rate=dropout))
model.add(Bidirectional(LSTM(window_size, return_sequences=False))) 
model.add(Dense(units=1))
model.add(Activation('linear'))

batch_size = 16
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, shuffle=False, validation_split=0.2)

y_pred = model.predict(x_test)
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)

plt.plot(y_test_orig, label='Actual Price', color='orange')
plt.plot(y_pred_orig, label='Predicted Price', color='green') 
plt.title('BTC Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend(loc='best')
plt.show()
  

