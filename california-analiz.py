import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn  as sbn
sbn.set()


cities = pd.read_csv("california_cities.csv")

print(cities.head())

#nitelik seçimi
latitude , longitude = cities["latd"] , cities["longd"]
population , area = cities["population_total"] , cities["area_total_km2"]

#grafik oluşturma 

plt.scatter(longitude, latitude , label = None , c = np.log10(population),
cmap = 'viridis', s = area , linewidth=0, alpha=0.5)

plt.axis( 'equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label = 'log$_{10}$(population)')
plt.clim(3,7)

#boş liste oluşturma 

for area in [100,300,500]:
    plt.scatter([],[],c = 'k',alpha=0.3,s = area , label = str(area)+'km$^2$')
plt.legend(scatterpoints = 1, frameon = False , labelspacing = 1 ,title = 'City areas')
plt.title("area and population of california cities")
plt.show()  



"""
# Şehirlerin su alanlarını sıralayarak en yüksek su alanına sahip şehirleri bulalım
top_water_area_cities = cities[['city', 'area_water_km2']].sort_values(by='area_water_km2', ascending=False).head(10)

# Toplam su alanını hesaplayalım
total_water_area = cities['area_water_km2'].sum()

# Su alanının toplam alana oranını hesaplayalım
cities['water_area_percentage'] = (cities['area_water_km2'] / cities['area_total_km2']) * 100

# İlk birkaç satırı göstererek verinin genel yapısını inceleyelim
water_area_percentage_overview = cities[['city', 'area_water_km2', 'area_total_km2', 'water_area_percentage']].sort_values(by='water_area_percentage', ascending=False).head(10)

print(top_water_area_cities), 
print(total_water_area),
print(water_area_percentage_overview)"""





