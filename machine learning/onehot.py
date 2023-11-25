# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn import linear_model

# Read the CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\house price_onehot.csv")

# Create dummy variables for the 'town' column using one-hot encoding
dummies = pd.get_dummies(df.town)

# Merge the original DataFrame with the dummy variables
merged = pd.concat([df, dummies], axis='columns')

# Drop the original 'town' column and one of the dummy columns ('west windsor') to avoid multicollinearity
final = merged.drop(['town', 'west windsor'], axis='columns')

# Create a Linear Regression model
reg = linear_model.LinearRegression()

# Prepare the feature matrix X (all columns except 'price') and the target variable y ('price')
X = final.drop(['price'], axis=1)
y = final.price

# Fit the Linear Regression model on the data
reg.fit(X, y)

# Make a price prediction for a given set of features
predict = reg.predict([[2000, 0, 1]])

# Print the predicted price
print("The price is", predict)
score=reg.score(X,y)
print(score)