import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("C:\\Users\\Hayfa\\Documents\\insurance.csv")
X_train,X_test,y_train,y_test=train_test_split(df[['age']],df.insurance,train_size=0.9)
reg=LogisticRegression()
reg.fit(X_train,y_train)
predict = reg.predict(X_test)
print("The age is ",X_test['age'].values)
print("predicted values for y are ",predict)
prob=reg.predict_proba(X_test)
print(prob)