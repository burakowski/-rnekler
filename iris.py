import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)



# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)





iris = pd.read_csv("Iris.csv")

print(iris.head())


print(iris["Species"].value_counts())

"""iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
plt.show()"""

"""sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
plt.show()"""


"""sns.FacetGrid(iris, hue="Species") \
.map(plt.scatter,"SepalLengthCm","SepalWidthCm")
plt.legend()
plt.show()"""


"""sns.boxplot(x="Species",y="PetalLengthCm",data=iris,color="blue")
plt.show()"""

"""ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
plt.show()"""


"""sns.violinplot(x="Species",y="PetalLengthCm",data=iris,color="red") ### yaprak grafiği
plt.show()"""

"""sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
plt.show()"""

"""fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()"""


####### sklearn tensorflow öğrendikten sonra ilersini de yap