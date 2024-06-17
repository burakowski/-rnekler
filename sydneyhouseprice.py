import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("SydneyHousePrices.csv")
print(df)
print(df.info())     # veri tabanının analizi ve boşluk olup olmaması
print(df.describe())
print(df.isnull().sum())

print(df.select_dtypes(["object"]).columns) #katagorik seçilim

print(df["suburb"].value_counts()) # mahalle
print(df["propType"].value_counts()) # ev türü villa terrace dublex...

print(df.describe().T)


df["Date"] = pd.to_datetime(df["Date"])  # tarih sütınunu datetime veri tipine çevirdik
df["Year"] = df["Date"].dt.year           # datetime sütınundan yıl-ay-gün bilgileri alınarak yeni sütüne çevirildi
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day


print(df.head())

df = df.drop(["Id","Date"],axis=1) # burda da ihtiyaç kalmayan date ve id sütununu veriden çıkardık

print(df)


list_name = []
list_type = []
list_total_value = []
list_missing_value = []
list_unique_value = []

for i in df.columns:
    list_name.append(i)
    list_type.append(str(df[i].dtype))
    list_total_value.append(df[i].notnull().sum())
    list_missing_value.append(df[i].isnull().sum())
    list_unique_value.append(len(df[i].unique()))

    df_info = pd.DataFrame(data={"Total_Value":list_total_value,"Missing_Value":list_missing_value,"Unique_Value":list_unique_value,"Type":list_type},index=list_name)



print(df.info())


#veri analiz edilip temize çekildi şimdi görselleştirme kısmına başlıyoruz


"""df["suburb"].value_counts()[:15].plot.barh() # yerleşim yeri bar şeklinde gösterdik
plt.show()"""

"""df["propType"].value_counts()[:15].plot.barh() # ev türlerini grafiği
plt.show()"""


"""data_num = df.select_dtypes(["float64","int64"]).columns
fig,ax=plt.subplots(nrows=4, ncols=2, figsize=(15,15))
count=0    # burda sayısal veri olan tüm sütunları tablolaştırdık hepsini bir arada yaptık
for i in range(4):
    for j in range(2):
        sns.kdeplot(df[data_num[count]], ax = ax[i][j], shade=True, color="#008080")
        count+=1"""
  
"""sns.countplot(df["Month"]) # aya göre satış verileri
plt.show()
"""

"""
plt.figure(figsize=(10,5))
sns.barplot(x = df["Year"], y = df["sellPrice"], data = df) # yıllara göre ev fiyatları
plt.show()"""

"""sns.barplot(x = df["Month"], y = df["sellPrice"], data = df) # Aylara göre ev fiyatları
plt.show()"""


"""heat = pd.pivot_table(data = df,
                    index = 'Month',
                    values = 'sellPrice',#Bu kod, bir DataFrame'deki verileri bir ısı haritasına (heatmap) dönüştürmek için kullanılır. İşlem adım adım şu şekildedir:
                    columns = 'Year')
heat.fillna(0, inplace = True)
print(heat)"""

"""plt.figure(figsize=(15,10))
plt.title('Yıllara ve Aylara Ev Fiyat Ortalamaları Isı Haritası')
sns.heatmap(heat)
"""


##burda kaldık https://www.kaggle.com/code/nafizcntz/sydney-ev-fiyatlar-tahmini-lineer-regresyon suburd değişkeni sınıflandırılması













