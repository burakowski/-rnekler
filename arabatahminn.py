import pandas as pd
import numpy as nb
import matplotlib.pyplot as plt
import seaborn as sbn


dataFrame = pd.read_excel("merc.xlsx")
print(dataFrame)   # verileri getir
print(dataFrame.head())   # ilk 5 veri
print(dataFrame.describe())   # veri bilgileri
print(dataFrame.isnull().sum())   # verilerde boş var mı diye kontrol ediyoruz

# veriyi analiz ettik boş var mı diye kontrol ettik
# grafiksel kısma geçiyoruz

"""plt.figure(figsize=(5,5))
sbn.displot(dataFrame["price"])  # fiyat grafiği
plt.show()"""


# çok uç kısımlarda olanlar yüzünden grafik uzuyor o yüzden onları çıkaracağız bu tamamen kişiye bağlı istemezsen çıkarma


"""sbn.countplot(dataFrame["year"])  # yıl grafiği
plt.show()"""

"""print(dataFrame.corr()["price"].sort_values())"""  # fiyatların diğer özelliaklerle ilişkisi negatif y da pozitif olarak


"""sbn.scatterplot(x="mileage", y="price",data=dataFrame)  ## gidilen km arttıkça fiyat düşüyor
plt.show()"""

"""sbn.scatterplot(x="engineSize",y="price",data=dataFrame) # motor gücü ile fiyat orantısı
plt.show()"""

print(dataFrame.sort_values("price",ascending=False).head(20))  # en yüksek fiyatı en yukarda getirir sıralama
print(dataFrame.sort_values("price",ascending=True).head(20))  # en düşük fiyatı en aşağıda getirir sıralama

print(len(dataFrame))
print(len(dataFrame) * 0.01)
# burda ise en yüksek 131 tane arabayı listeden atacağız grafiğin daha düzgün durması için tamamen kişisel tercih

"""print(dataFrame.sort_values("price",ascending=False).iloc[131:]) #burda yüksek fiyatlı 131 aracı sildik"""
pahaliaraclarsizDf = dataFrame.sort_values("price",ascending=False).iloc[131:]
print(pahaliaraclarsizDf)

"""plt.figure(figsize=(7,5))
sbn.displot(pahaliaraclarsizDf["price"])  #güncellenmiş fiyat grafiği eski verinin %99
plt.show()"""

print(dataFrame.groupby("year").mean("price"))# yıllara göre fiyat ortalaması
print(pahaliaraclarsizDf.groupby("year").mean("price"))# yıllara göre fiyat ortalaması ama pahalı arabalar çıkarılmış


print(dataFrame[dataFrame.year != 1970].groupby("year").mean("price")) # 1970 fiyatları çıkarılmış halde


dataFrame = pahaliaraclarsizDf  #artık kendi datamızdan da pahalı araçları çıkardık
dataFrame = dataFrame[dataFrame.year != 1970] # artık 1970 verilerinide çıkardık
print(dataFrame.groupby("year").mean("price"))  # temiz halde verilerimiz

#şimdi tranmission kısmını atacağız çünkü hepsi aynı boş veri kaplıyor

dataFrame = dataFrame.drop("transmission",axis=1)
print(dataFrame)

# veriyi temizledik şimdi model oluşturma kısmına geçiyoruz


y = dataFrame["price"].values # ulaşmak istediğimiz veri bu numby dizisine çevirdik
x = dataFrame.drop("price",axis=1).values # tüm özellikler yani featruing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)

print(len(x_train))  # verimizi 30 a 70 olarak böldük
print(len(x_test))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)    # özellikleri scaler(ölçeklendirme) ettik fiyatı yani y yi etmemize gerek yok
x_train = scaler.transform(x_train)


from  tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential



print(x_train.shape)


model = Sequential() #model oluşturduk
model.add(Dense(12,activation="relu")) #burda modele 4 tane  katman ekledik  ve her birine 12 adet nöron ekledik deneme yanılma ile
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))  #çıkış katmanı buna activation eklemeye gerek yok

model.compile(optimizer="adam",loss="mse")  #model oluşturuldu


model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300) #model eğitme ve validution_data 100 de 30 luk olan test kısmı ile kıyaslamaya yarar batch ise part part veriyoruz veriyi modele yormamak için

kayip_Verisi = pd.DataFrame(model.history.history)

print(kayip_Verisi.head())

kayip_Verisi.plot()   #tahmin ile olanı kıyasladık
plt.show()


from sklearn.metrics import mean_absolute_error,mean_squared_error

tahmindizisi = model.predict(x_test)  # predict tahmin yapar
print("tahmindizisi" )

mean_absolute_error(y_test, tahmindizisi)


print(dataFrame.describe())

plt.scatter(y_test, tahmindizisi)
plt.plot(y_test,y_test,"g-*")      #tahmin fiyatı ile olanı kıyasladık
plt.show()


yeniarabaseries = dataFrame.drop("price",axis=1).iloc[2]  #burada 2.arabayı listeden çıkartıp tekrar ekledik ve fiyat tahmmini yaptırdık

yeniarabaseries = scaler.transform(yeniarabaseries.values.reshape(-1,5))

model.predict(yeniarabaseries)

