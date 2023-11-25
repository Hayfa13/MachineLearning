import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

# Read the CSV file
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\tested.csv")

# Define input features and target
inputs = df[['Pclass', 'Sex', 'Age', 'Fare']].copy()  # Create a copy of the DataFrame
target = df['Survived']

# Create a LabelEncoder for the 'Sex' column
le_sex = LabelEncoder()
inputs['sex_n'] = le_sex.fit_transform(inputs['Sex'])
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
# Drop the original 'Sex' column
input_n = inputs.drop(['Sex'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_n, target, test_size=0.2)

# Create a DecisionTreeClassifier and fit it to the training data
reg = tree.DecisionTreeClassifier()
reg.fit(X_train, y_train)

# Calculate the accuracy score on the testing data
score = reg.score(X_test, y_test)
print("Accuracy Score:", score)
