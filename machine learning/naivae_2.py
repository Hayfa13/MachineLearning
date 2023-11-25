import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
wines=load_wine()
print(dir(wines))
print(wines.feature_names)
print(wines.target_names)
df=pd.DataFrame(wines.data, columns=wines.feature_names)
print(df.head())
print()
df['target']= wines.target
X_train, X_test, y_train, y_test = train_test_split(wines.data, wines.target, test_size=0.3, random_state=100)# to increase randomization

reg=GaussianNB()
reg.fit(X_train,y_train)
print("The score of Gaussian is ",reg.score(X_test,y_test))

mn=MultinomialNB()
mn.fit(X_train,y_train)
print("The score of Gaussian is ",mn.score(X_test,y_test))