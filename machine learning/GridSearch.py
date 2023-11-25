import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Load the Iris dataset, which is a popular dataset for classification problems
iris = load_iris()

# Create a DataFrame from the Iris dataset, with columns named after feature names
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add a 'flower' column to the DataFrame, which contains the target labels (species)
# Also, replace numeric labels with their corresponding target names for better readability
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])

# Create a GridSearchCV object to perform hyperparameter tuning for an SVM classifier
clf = GridSearchCV(SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['linear', 'rbf']
}, cv=5, return_train_score=False)

# Fit the GridSearchCV object on the Iris data to find the best hyperparameters
clf.fit(iris.data, iris.target)

# Get the results of the cross-validation grid search
cv_results = clf.cv_results_

# Create a DataFrame from the cross-validation results
df = pd.DataFrame(cv_results)

# Print the relevant columns from the DataFrame, including parameter values and mean test scores
print(df[['param_C', 'param_kernel', 'mean_test_score']])

# Print the best mean test score found during hyperparameter tuning
print("Best score is ", clf.best_score_)

# Print the best hyperparameters that achieved the best mean test score
print("Best param is ", clf.best_params_)



