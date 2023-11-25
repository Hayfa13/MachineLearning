import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\heart.csv")
print(df.describe())

# Remove outliers based on Cholesterol, Oldpeak, and RestingBP
df1 = df[df['Cholesterol'] <= (df['Cholesterol'].mean() + 3 * df['Cholesterol'].std())]
df2 = df1[df1['Oldpeak'] <= (df1['Oldpeak'].mean() + 3 * df1['Oldpeak'].std())]
df3 = df2[df2['RestingBP'] <= (df2['RestingBP'].mean() + 3 * df2['RestingBP'].std())]
df4 = df3.copy()

# Replace categorical values with numeric values
df4['ExerciseAngina'].replace({'N': 0, 'Y': 1}, inplace=True)
df4['ST_Slope'].replace({'Down': 1, 'Flat': 2, 'Up': 3}, inplace=True)
df4['RestingECG'].replace({'Normal': 1, 'ST': 2, 'LVH': 3}, inplace=True)

print(df4.head())

# Perform one-hot encoding for categorical variables
df5 = pd.get_dummies(df4, drop_first=True)

# Separate features (X) and target variable (y)
X = df5.drop("HeartDisease", axis='columns')
y = df5['HeartDisease']

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=10)

# Train RandomForestClassifier on the original data
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("The rf score is ", rf.score(X_test, y_test))

# Apply PCA to retain 95% of information
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)

# Split the data with PCA components into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# Train RandomForestClassifier on PCA-transformed data
rf_pca = RandomForestClassifier()
rf_pca.fit(X_train_pca, y_train_pca)
print("The rf score on PCA-transformed data is ", rf_pca.score(X_test_pca, y_test_pca))
