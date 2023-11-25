import pandas as pd
import numpy as np
from sklearn import linear_model
import math
from word2number import w2n #to chnage word to numerics
df=pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\hiring.csv")

df.experience.fillna('zero',inplace=True)#first preprocessing
df['exp_age'] = df['experience'].apply(w2n.word_to_num) #converting to number
median = math.floor(df['test_score(out of 10)'].median())
df['test_score(out of 10)'].fillna(median, inplace=True)


reg=linear_model.LinearRegression()
reg.fit(df[['test_score(out of 10)','interview_score(out of 10)','exp_age']],df.salary)
prediction=reg.predict([[9,6,2]])
print("The salary is: ",prediction)