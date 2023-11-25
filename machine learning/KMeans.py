import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())
df['flower']= iris.target
df.drop(['sepal length (cm)','sepal width (cm)','flower'],axis=1,inplace=True)
km=KMeans(n_clusters=3)
y_predicted = km.fit_predict(df)
df['cluster']= y_predicted

df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='yellow')
plt.show()

sse=[]
k_rng = range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()