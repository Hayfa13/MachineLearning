import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset (make sure to specify the correct file path)
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\HR_comma_sep.csv")

# Select relevant features (columns) for analysis
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]

# Separate the data into 'left' and 'retained' dataframes
left = df[df.left == 1]
retained = df[df.left == 0]

# Create dummy variables for the 'salary' column
salary_dummies = pd.get_dummies(subdf.salary, prefix='salary')
df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
df_with_dummies.drop('salary', axis='columns', inplace=True)

# Prepare the feature matrix (X) and target variable (y)
X = df_with_dummies
y = df.left

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Create a Logistic Regression model
reg = LogisticRegression()

# Fit the model to the training data
reg.fit(X_train, y_train)

# Make predictions on the test data
predictions = reg.predict(X_test)

# Print the feature values of X in the test set
print("Feature values of X in the test set:")
print(X_test)

# Print the predicted values (y) on the test set
print("Predicted values (y) on the test set:")
print(predictions)

# Calculate and print the accuracy score of the model on the training data
score = reg.score(X_train, y_train)
print("Model accuracy on training data:", score)
