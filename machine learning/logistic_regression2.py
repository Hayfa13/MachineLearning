import pandas as pd
from matplotlib import pyplot as plt

# Load the dataset (make sure to specify the correct file path)
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\HR_comma_sep.csv")

# Display the first few rows of the dataset
print(df.head())

# Separate the data into 'left' and 'retained' dataframes
left = df[df.left == 1]
retained = df[df.left == 0]

# Create a bar plot for salary vs. left
pd.crosstab(df.sales, df.left).plot(kind='bar')
plt.xlabel('Salary')
plt.ylabel('Count')
plt.title('Employee Retention by Salary')
plt.show()
