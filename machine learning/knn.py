import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import classification_report
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())

df0 = df[df['target'] == 0]
df1 = df[df['target'] == 1]
df2 = df[df['target'] == 2]

X = df.drop(['target', 'flower'], axis=1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

y_predicted =knn.predict(X_test)
cm=confusion_matrix(y_test,y_predicted)

print(classification_report(y_test,y_predicted))

plt.figure(figsize=(7,5))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='red', label=iris.target_names[0])
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='green', label=iris.target_names[1])
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color='blue', label=iris.target_names[2])

plt.title("Scatter Plot of Sepal Length vs. Sepal Width")
plt.show()