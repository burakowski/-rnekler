import pandas as pd 
import numpy as np 

# CSV dosyalarını oku
movie = pd.read_csv('movie.csv')
rating = pd.read_csv('rating.csv').drop("timestamp", axis=1)

# İlk birkaç satırı göster
print(movie.head())
print(rating.head())

# Veri tiplerini göster
print(movie.dtypes)
print(rating.dtypes)

# movieId ve userId sütunlarını object (string) tipine dönüştür
movie["movieId"] = movie["movieId"].astype(object)
rating["movieId"] = rating["movieId"].astype(object)
rating["userId"] = rating["userId"].astype(object)

# Eksik değerlerin sayısını göster
print(movie.isnull().sum())

# Verileri üzerinde değişikliğe gitmeden önce tüm oyların ortalamasını alalım.
C = rating['rating'].mean()
print(C)


#Bu kod bloğu, rating DataFrame'inde her film için kaç tane oy (rating) verildiğini sayar ve ardından sonuçları düzenler.
rating_count = rating.groupby(["movieId"]).count()
rating_count["movieId"] = rating_count.index
rating_count.index.name = None
rating_count.reset_index(inplace=True)
del rating_count["index"]
print(rating_count)

# Şimdi de rating verisini film Id’lerine göre ortalamalarını veren yeni bir veri ortaya koyuyoruz. Daha sonradan bu veri üzerinde de index düzenlemeleri yapıyoruz.
means = rating.groupby(["movieId"]).mean()
means["movieId"] = means.index
means.index.name = None
means.reset_index(inplace=True)
del means["index"]
means = means.rename(columns={"rating": "mean"})
print(means)

# Elde ettiğimiz iki veriyi merge modülü movieId değişkenine göre birleştirelim.
df = pd.merge(movie, rating_count)
df = pd.merge(df, means)
df.head()

# kullanıcı sayısının %90 dan fazla oy alan listeler
m = df["userId"].quantile(0.90)
print(m)

# Oylama sayısına göre uygun olan filmler
q_movies = df.copy().loc[df["userId"] >= m]
print(q_movies.shape)

# Veri setinde 2765 adet film yer almaktadır. Her film için aynı metriğin hesaplanması gerekir. Daha sonra yeni bir score özelliği tanımlayıp, veri setindeki verileri bu özelliğe göre sıralıyoruz.
def weighted_rating(x, m=m, C=C):
    v = x["userId"]
    R = x["mean"]
    return (v / (v + m) * R) + (m / (m + v) * C)

q_movies["score"] = q_movies.apply(weighted_rating, axis=1)

# Verileri score sütununa göre azalan sıralıyoruz
q_movies = q_movies.sort_values("score", ascending=False)
print(q_movies.head(10))
