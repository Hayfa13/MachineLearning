import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm

# Read the CSV file into a DataFrame
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\weight-height.csv")

# Plotting histogram of height
plt.figure()
plt.hist(df['Height'], bins=20, width=0.8)  # 'width' instead of 'riwdth'
plt.xlabel("Height (inches)")
plt.ylabel("Count")


# Displaying descriptive statistics for height
print(df['Height'].describe())

# Plotting histogram of height with density
plt.figure()
plt.hist(df['Height'], bins=20, width=0.8, density=True)  # 'width' instead of 'riwdth'
plt.xlabel("Height (inches)")
plt.ylabel("Count")

# Generating a normal distribution curve
rng = np.arange(df['Height'].min(), df['Height'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, df['Height'].mean(), df['Height'].std()))

# Calculating upper and lower limits using z-score
upper_limit = df['Height'].mean() + 3 * df['Height'].std()
lower_limit = df['Height'].mean() - 3 * df['Height'].std()
print("Upper Limit:", upper_limit)
print("Lower Limit:", lower_limit)

# Displaying samples beyond the upper and lower limits
outliers = df[(df['Height'] > upper_limit) | (df['Height'] < lower_limit)]
print("Outliers:")
print(outliers)

# Creating a new DataFrame without outliers
df_filtered = df[(df['Height'] < upper_limit) & (df['Height'] > lower_limit)]
print("Filtered DataFrame:")
print(df_filtered)
print("Filtered DataFrame Shape:", df_filtered.shape)

# Calculating z-score for each height entry
df['z_score'] = (df['Height'] - df['Height'].mean()) / df['Height'].std()

# Displaying entries with z-score greater than 3 or less than -3
print("Entries with z-score > 3:")
print(df[df['z_score'] > 3])
print("Entries with z-score < -3:")
print(df[df['z_score'] < -3])

# Creating a new DataFrame without z-score outliers
df_no_zscore_outliers = df[(df['z_score'] > -3) & (df['z_score'] < 3)]
print("DataFrame without z-score outliers:")
print(df_no_zscore_outliers)
print("DataFrame without z-score outliers Shape:", df_no_zscore_outliers.shape)

plt.show()