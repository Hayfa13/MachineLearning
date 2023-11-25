import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\homeprices_multi.csv")
median = math.floor(df.bedrooms.median()) #to find the median of bedroom to fill na spaces and rounding it off
df['bedrooms'].fillna(median, inplace=True) #filling na places

reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)

prediction = reg.predict([[3000, 3, 40]])
print("The price is", prediction)
