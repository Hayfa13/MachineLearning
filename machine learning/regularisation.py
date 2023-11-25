import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
# Load the dataset
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\melb_data.csv")
print(df.head())
df['BuildingArea'] = pd.to_numeric(df['BuildingArea'], errors='coerce')

# Select relevant columns
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount',
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
df = df[cols_to_use]
print(df.isna().sum())

# Fill missing values with zeros for selected columns
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)
print(df.isna().sum())

# Fill missing values with mean for specific columns
df['Landsize'] = df['Landsize'].fillna(df['Landsize'].mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].mean())
print(df.isna().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Perform one-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)
print()
print(df.head())

# Split the data into features (X) and target variable (y)
x = df.drop('Price', axis=1)
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

ridge_reg = linear_model.Ridge(alpha=100,max_iter=200,tol=0.5)
ridge_reg.fit(X_train,y_train)
print("The score of Linear Regression (test) ", ridge_reg.score(X_test, y_test))
print("The score of Linear Regression (train) ", ridge_reg.score(X_train, y_train))
