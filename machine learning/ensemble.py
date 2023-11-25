import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\diabetes.csv")

# Check for missing values
print(df.isnull().sum())

# Display summary statistics
print(df.describe())

# Check the distribution of the 'Outcome' variable
print(df['Outcome'].value_counts())

# Separate features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)

# Train a Decision Tree model
reg = DecisionTreeClassifier()
scores = cross_val_score(reg, X, y, cv=5)
print("Decision Tree Cross-Validation Mean Score:", scores.mean())

# Train a Bagging Classifier with Decision Tree base estimator
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,#how many datas its split into
    max_samples=0.8,#data that is selected in different sets , 0.2 not selected for any set which is used for training the model
    oob_score=True
)
bag_model.fit(X_train, y_train)
print("Bagging Classifier OOB Score:", bag_model.oob_score_)

# Cross-Validation with Bagging Classifier
scores_bag = cross_val_score(bag_model, X, y, cv=5)
print("Bagging Classifier Cross-Validation Scores:", scores_bag)

# Cross-Validation with RandomForest Classifier
scores_rf = cross_val_score(RandomForestClassifier(), X, y, cv=5)
print("RandomForest Classifier Cross-Validation Mean Score:", scores_rf.mean())

