# Import necessary libraries
#for preprocessing two libraries extra imported
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Read the CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\house price_onehot.csv")

# Create a LabelEncoder to encode the 'town' column
le = LabelEncoder()
df['town_encoded'] = le.fit_transform(df['town'])

# Create a OneHotEncoder for the 'town_encoded' column
ohe = OneHotEncoder(sparse=False, drop='first')

# Transform the 'town_encoded' column into one-hot encoded features
X_encoded = ohe.fit_transform(df[['town_encoded']])

# Concatenate the one-hot encoded features with the 'area' column
X = np.column_stack((X_encoded, df['area']))

# Prepare the target variable y ('price')
y = df['price']

# Create a Linear Regression model
reg = linear_model.LinearRegression()

# Fit the Linear Regression model on the data
reg.fit(X, y)

# Make a price prediction for a given set of features
predict = reg.predict([[0,1, 2000]])  # Example: 'town' is the first category (encoded as 0), 'area' is 2000
print("The price is", predict)

# Calculate the R-squared score of the model
score = reg.score(X, y)
print("R-squared score:", score)
