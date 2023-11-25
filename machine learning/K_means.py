import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset from a CSV file
df = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\salary.csv")

# Apply Min-Max scaling to 'Income' and 'Age'
scaler = MinMaxScaler()
df[['Income', 'Age']] = scaler.fit_transform(df[['Income', 'Age']])

# Create a KMeans clustering model with 3 clusters
km = KMeans(n_clusters=3)

# Fit and predict the clusters based on 'Age' and 'Income' features
y_predicted = km.fit_predict(df[['Age', 'Income']])  # 'Age' and 'Income' should be in a list, not separate lists

# Add the cluster labels to the DataFrame
df['cluster'] = y_predicted

# Display the first few rows of the DataFrame with cluster labels
print(df.head())

# Separate data for each cluster
df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

# Visualize the clusters with scaled data
plt.scatter(df0['Age'], df0['Income'], color='green')
plt.scatter(df1['Age'], df1['Income'], color='red')
plt.scatter(df2['Age'], df2['Income'], color='black')
print(km.cluster_centers_)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='yellow',marker='*')
plt.xlabel('Age')  # Set the x-axis label
plt.ylabel('Income')  # Set the y-axis label
plt.show()  # Display the scatter plot with scaled clustered data

k_rng=range(1,10)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['Age','Income']])
    sse.append(km.inertia_)
    print(sse)

plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(k_rng,sse)
plt.show()
