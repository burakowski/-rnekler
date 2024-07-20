import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Enflasyon oranlarını içeren sözlük (yıllık)
inflation_rates = {
    2000: 0.03, 2001: 0.02, 2002: 0.01, 2003: 0.02, 2004: 0.03,
    2005: 0.03, 2006: 0.02, 2007: 0.02, 2008: 0.04, 2009: 0.01,
    2010: 0.02, 2011: 0.03, 2012: 0.02, 2013: 0.01, 2014: 0.02,
    2015: 0.01, 2016: 0.02, 2017: 0.02, 2018: 0.03, 2019: 0.02,
    2020: 0.01, 2021: 0.04, 2022: 0.03, 2023: 0.02
}

# Enflasyon etkisini fiyatlara uygulama fonksiyonu
def adjust_for_inflation(price, year, current_year=2023):
    inflation_adjusted_price = price
    for y in range(year, current_year):
        inflation_adjusted_price *= (1 + inflation_rates.get(y, 0))
    return inflation_adjusted_price

# Veri Yükleme ve Ön İşleme
dataFrame = pd.read_excel("merc.xlsx")
print(dataFrame.head())
print(dataFrame.describe())
print(dataFrame.isnull().sum())

# Enflasyon etkisini fiyatlara uygulama
dataFrame['adjusted_price'] = dataFrame.apply(lambda row: adjust_for_inflation(row['price'], row['year']), axis=1)

plt.figure(figsize=(5,5))
sbn.displot(dataFrame["adjusted_price"])  # fiyat grafiği
plt.show()

# Pahalı Araçların Çıkarılması
pahaliaraclarsizDf = dataFrame.sort_values("adjusted_price", ascending=False).iloc[131:]
dataFrame = pahaliaraclarsizDf[pahaliaraclarsizDf.year != 1970]
dataFrame = dataFrame.drop("transmission", axis=1)

plt.figure(figsize=(7,5))
sbn.displot(pahaliaraclarsizDf["adjusted_price"])  # güncellenmiş fiyat grafiği
plt.show()

sbn.scatterplot(x="mileage", y="adjusted_price", data=dataFrame)  ## gidilen km arttıkça fiyat düşüyor
plt.show()

sbn.scatterplot(x="engineSize", y="adjusted_price", data=dataFrame) # motor gücü ile fiyat orantısı
plt.show()

# Model İçin Veriyi Hazırlama
y = dataFrame["adjusted_price"].values
x = dataFrame.drop(["price", "adjusted_price"], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model Oluşturma ve Eğitim
model = Sequential()
model.add(Dense(12, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(12, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(12, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(12, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=250, epochs=300, callbacks=[early_stopping])

# Eğitim Kayıpları
kayip_Verisi = pd.DataFrame(history.history)
kayip_Verisi.plot()
plt.show()

# Tahmin ve Değerlendirme
tahmindizisi = model.predict(x_test)
print(mean_absolute_error(y_test, tahmindizisi))
print(dataFrame.describe())

plt.scatter(y_test, tahmindizisi)
plt.plot(y_test, y_test, "g-*")
plt.show()

# Yeni Araç Bilgileriyle Tahmin
yeniarabaseries = dataFrame.drop(["price", "adjusted_price"], axis=1).iloc[2]  # Burada 2.arabayı listeden çıkartıp tekrar ekledik ve fiyat tahminini yaptırdık
yeniarabaseries = scaler.transform(yeniarabaseries.values.reshape(-1, 5))
predicted_price = model.predict(yeniarabaseries)
print(f"Tahmin edilen fiyat: {predicted_price[0][0]:.2f} euro")

# Kullanıcıdan alınan özelliklerle fiyat tahmini yapan fonksiyon
def predict_custom_car_price(year, mileage, tax, mpg, engineSize):
    custom_car = np.array([[year, mileage, tax, mpg, engineSize]])
    custom_car_scaled = scaler.transform(custom_car)
    predicted_price = model.predict(custom_car_scaled)
    return predicted_price[0][0]

# Örnek kullanım
year = 2018
mileage = 15000
tax = 150
mpg = 40.0
engineSize = 2.0

predicted_price = predict_custom_car_price(year, mileage, tax, mpg, engineSize)
print(f"Özellikleri verilen aracın tahmin edilen fiyatı: {predicted_price:.2f} euro")

