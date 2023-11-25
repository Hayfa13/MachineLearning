import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\height.csv")

# Displaying descriptive statistics for the DataFrame
print(df.describe())

# Calculating the first and third quartiles
Q1 = df['height'].quantile(0.25)
Q3 = df['height'].quantile(0.75)
print("Q1:", Q1, "Q3:", Q3)

# Calculating the interquartile range (IQR)
IQR = Q3 - Q1

# Calculating upper and lower limits for outlier detection
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR
print("Upper Limit:", upper_limit, "Lower Limit:", lower_limit)

print(df[(df['height'] < lower_limit) | (df['height'] > upper_limit)])

# Creating a new DataFrame without outliers
df_filtered = df[(df['height'] > lower_limit) & (df['height'] < upper_limit)]
print("DataFrame without outliers:")
print(df_filtered)
