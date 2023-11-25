import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris=load_iris()

df= pd.DataFrame(iris.data , columns=iris.feature_names) #feature names = seal length sepal width ..
print(df.head())

df['target']= iris.target #type of flower
print()
print(df.head())

df[df.target==1].head()
df['flower_name']= df.target.apply(lambda x: iris.target_names[x])
print()
print(df.head())
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green', marker='*')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue', marker='*')
plt.show()

X=df.drop(['target','flower_name'],axis=1)
y=df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
reg=SVC(C=10)#gamma ,kernel,linear
reg.fit(X_train,y_train)
score=reg.score(X_test,y_test)
print("The score is ",score)