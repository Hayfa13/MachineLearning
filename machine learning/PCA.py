import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

digits = load_digits()

# Print features and data for the first sample
print(digits.feature_names)
print(digits.data[0])

# Create a DataFrame from the dataset
df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
df['target'] = digits.target

print(df.head())
print()
print(df.describe())

# Separate features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("X_scaled values", X_scaled)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

# Train Logistic Regression model on the original data
reg = LogisticRegression()
reg.fit(X_train, y_train)
print()
print("Accuracy on original data:", reg.score(X_test, y_test))

# Apply PCA to retain 95% of information
pca = PCA(0.95)
X_pca = pca.fit_transform(X_scaled)

print(X_pca)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Number of components:", pca.n_components_)

# Split the dataset with PCA components into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=30)

# Train Logistic Regression model on the PCA-transformed data
reg_pca = LogisticRegression(max_iter=1000)
reg_pca.fit(X_train_pca, y_train_pca)
print()
print("Accuracy on PCA-transformed data:", reg_pca.score(X_test_pca, y_test_pca))

# Display the first digit as an image
plt.gray()
plt.matshow(digits.images[0])
plt.show()