#importing numpy,pandas matplotlib for data
#sklearn -> linear regression model imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

df=pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\House Price.csv")#df -> dataframe in pandas , reading CSV file
print(df)#data printed
plt.xlabel("Area(sq.ft)")
plt.ylabel("Price(Rs)")
plt.scatter(df['area'], df['price'],color='blue',marker='*')#x-axis area , y-axis price

reg = linear_model.LinearRegression()#calling an object of the class
reg.fit(df[['area']],df.price)#fitting the model

areas=np.array(df[['area']])#area stored as array
plt.plot(areas,reg.predict(areas),color='red',label='Best Fit Line')


d=pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\areas.csv")
p=reg.predict(d[['area']])
d['prices'] = p
d.to_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\areas.csv")

with open('model_pickle','wb') as f:
    pickle.dump(reg,f)
with open('model_pickle','rb') as f:
    mp=pickle.load(f)
predict =mp.predict([[5000]])
print("Predicted value is ",predict)
plt.show()