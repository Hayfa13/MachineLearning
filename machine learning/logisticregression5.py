import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import seaborn as sb

data=load_iris()
print(dir(data))
print(data.data)
print(data.target)
print(data.target_names)
X_train , y_train ,X_test ,y_test=train_test_split(data.data,data.target,train_size=0.8)
reg=LogisticRegression()
reg.fit(X_train,y_train)
reg.predict()