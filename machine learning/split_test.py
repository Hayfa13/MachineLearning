# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\carprice.csv")

# Define the features (X) and the target (y)
X = df[['mileage', 'age']]
y = df.price

# Initialize a Linear Regression model
reg = LinearRegression()

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the Linear Regression model to the training data
reg.fit(X_train, y_train)

# Predict the target values for the test set
predict_y = reg.predict(X_test)

# Print the test features (X values)
print("Test features (X values):")
print(X_test)

# Print the predicted y values
print("Predicted y values are:", predict_y)

# Calculate the R^2 score, a measure of the model's performance on the test data
score = reg.score(X_test, y_test)

# Print the R^2 score
print("R^2 score is:", score)
