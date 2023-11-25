# Importing necessary libraries
import pandas as pd

# Reading the CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\height.csv")

# Calculating the maximum threshold using the 95th percentile
max_threshold = df['height'].quantile(0.95)
print("Maximum Threshold:", max_threshold)

# Displaying samples exceeding the maximum threshold
print("Samples above Maximum Threshold:")
print(df[df['height'] > max_threshold])

# Calculating the minimum threshold using the 5th percentile
min_threshold = df['height'].quantile(0.05)
print("\nMinimum Threshold:", min_threshold)

# Displaying samples below the minimum threshold
print("Samples below Minimum Threshold:")
print(df[df['height'] < min_threshold])

# Selecting samples within the defined threshold range
filtered_df = df[(df['height'] < max_threshold) & (df['height'] > min_threshold)]
print("\nFiltered DataFrame:")
print(filtered_df)



