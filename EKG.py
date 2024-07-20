import pandas as pd 
import numpy as np

train = pd.read_csv("mitbih_train.csv")
x_train = np.array(train)[:,:187]
y_train = np.array(train)[:,187]


test = pd.read_csv("mitbih_test.csv")
x_test = np.array(train)[:,:187]
y_test = np.array(train)[:,187]


from sklearn.naive_bayes import CategoricalNB
gnb = CategoricalNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 
cm = confusion_matrix(y_test, y_pred,)
index = ['No','S','V','F','Q']
column = ['No','S','V','F','Q']

cm_df = pd.DataFrame(cm,column,index)
plt.figure(figsize=(10,6))
sns.heatmap(cm_df,annot= True, fmt="d")

from sklearn import metrics
print("Accuracy :",metrics.accuracy_score(y_test,y_pred))
