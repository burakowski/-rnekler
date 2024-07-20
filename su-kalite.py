import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Veriyi yükleme
data = pd.read_csv("sukalite.csv")

# İlk 5 satırı yazdırma
print(data.head(5))

# Verisetinde boş değer içeren tüm satırlar kaldırılmıştır.
data = data.dropna()
data.isnull().sum()

# Potability sütununda 0 ve 1 dağılımını görme
plt.figure(figsize=(15, 10))
sns.countplot(data.Potability)
plt.title("Distribution of Unsafe and Safe Water")
plt.show()

import plotly.express as px
data = data

# PH sütununa bakarak başlayalım
figure = px.histogram(data, x="ph", color="Potability", title="Factors Affecting Water Quality: PH")
figure.show()

# Hardness sütununu gösterme
figure = px.histogram(data, x="Hardness", color="Potability", title="Factors Affecting Water Quality: Hardness")
figure.show()

# Solids sütununu gösterme
figure = px.histogram(data, x="Solids", color="Potability", title="Factors Affecting Water Quality: Solids")
figure.show()

# Chloramines sütununu gösterme
figure = px.histogram(data, x="Chloramines", color="Potability", title="Factors Affecting Water Quality: Chloramines")
figure.show()


import plotly.express as px

figure = px.scatter(data, x="Hardness", y="ph", color="Potability", title="Relationship between Hardness and PH by Potability")
figure.show()




