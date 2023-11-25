import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the Titanic dataset from a CSV file
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\titanic.csv")

# Remove unnecessary columns from the DataFrame
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis=1, inplace=True)

# Display the first few rows of the DataFrame
print(df.head())

# Separate the target variable (Survived) and input features
target = df['Survived']
inputs = df.drop('Survived', axis=1)

# Create dummy variables for the 'Sex' column
dummies = pd.get_dummies(inputs['Sex'])
inputs = pd.concat([inputs, dummies], axis=1)
inputs.drop('Sex', axis=1, inplace=True)

# Fill missing values in the 'Age' column with the mean age
inputs['Age'] = inputs['Age'].fillna(inputs['Age'].mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

# Create a Gaussian Naive Bayes classifier
reg = GaussianNB()

# Train the classifier on the training data
reg.fit(X_train, y_train)

# Evaluate the classifier on the testing data and calculate the accuracy
score = reg.score(X_test, y_test)
print("The score is", score)

# Display the first 10 rows of the testing data and the corresponding true labels and predicted labels
print()
print(X_test[:10])
print()
print(y_test[:10])
print("Predicted is", reg.predict(X_test[:10]))
print(reg.predict_proba(X_test[:10]))