import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\car.csv")

# Encode the 'model' column using LabelEncoder
le = LabelEncoder()
df['mod_car'] = le.fit_transform(df['model'])

# Initialize a OneHotEncoder with drop='first' to create one-hot encoded features
ohe = OneHotEncoder(sparse=False, drop='first')

# Transform the 'mod_car' column into one-hot encoded features
mod_car_encoded = ohe.fit_transform(df[['mod_car']])

# Create a DataFrame for the one-hot encoded features
mod_car_encoded_df = pd.DataFrame(mod_car_encoded, columns=ohe.get_feature_names_out(['mod_car']))

# Concatenate the one-hot encoded features with 'mileage' and 'age' columns
X = pd.concat([mod_car_encoded_df, df[['mileage', 'age']]], axis=1)
print(X)
# Target variable
y = df.price

# Initialize the linear regression model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)
predict=reg.predict([[0,1,45000,4]])
print("The price is ",predict)
score=reg.score(X,y)
print(score)
# Scatter plot
plt.scatter(X['mod_car_1'], y, color="blue", marker='*')
plt.show()
