import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Read the CSV file
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\decisiontree.csv")

# Define inputs and target
inputs = df.drop('salary_more_than_100k', axis=1)  # Use axis=1 to specify column dropping
target = df['salary_more_than_100k']

# Create LabelEncoders
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

# Apply LabelEncoders to the respective columns
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

# Drop the original categorical columns
input_n = inputs.drop(['company', 'job', 'degree'], axis=1)  # Use axis=1 to specify column dropping

# Create a DecisionTreeClassifier and fit it to the data
reg = tree.DecisionTreeClassifier()
reg.fit(input_n, target)

# Make a prediction
predicted = reg.predict([[2, 2, 1]])
print("Google sales executive masters:", predicted)
