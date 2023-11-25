import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import pickle
import json
import matplotlib

# Data cleaning
df1 = pd.read_csv("C:\\Users\\Hayfa\\Documents\\machine learning\\data\\Bengaluru_House_Data.csv")
print(df1.head())
print(df1.shape)
print(df1.groupby('area_type')['area_type'].agg('count'))

df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)
print("\nAfter dropping columns")
print(df2.head())
print(df2.isnull().sum())

# Handling missing values
df3 = df2.dropna()
print(df3['size'].unique())

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
print(df3.bhk.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_sqft_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_num)
print(df4.head())
print(df4.loc[30])

# Feature engineering
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
print(df5.head())
print(df5.location.unique())
df5.to_csv("bhp.csv",index=False)
# Bringing in other category
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
print(location_stats)
location_stats_less_than_10 = location_stats[location_stats <= 10]
print(location_stats_less_than_10)
print(len(df5.location.unique()))
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))
print(df5.head())

# Outlier removal
# 300 sqft per bedroom as threshold criteria
print(df5[df5.total_sqft / df5.bhk < 300].head())
print(df5.shape)
df6 = df5[~(df5.total_sqft / df5.bhk < 300)]
print(df6.shape)


# Removing extreme cases using standard deviation
# Filtering beyond standard deviation
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df])
    return df_out
df7 = remove_pps_outliers(df6)
print(df7.shape)


# Same area same bedroom price difference
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]

    if bhk2.empty and bhk3.empty:
        print(f"No data points for 2 BHK and 3 BHK in {location}")
        return

    matplotlib.rcParams['figure.figsize'] = (15, 10)

    # Create a new figure for each plot
    plt.figure()

    if not bhk2.empty:
        plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50, alpha=0.5)

    if not bhk3.empty:
        plt.scatter(bhk3.total_sqft, bhk3.price, marker='*', color='green', label='3 BHK', s=50, alpha=0.5)

    plt.xlabel("Total sqft area")
    plt.ylabel("Price per sqft")
    plt.title(location)
    plt.legend()



plot_scatter_chart(df7, "Yeshwanthpur")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outliers(df7)
print(df8.shape)

plot_scatter_chart(df8,"Rajaji Nagar")

# Plotting histograms
matplotlib.rcParams["figure.figsize"] = (20, 10)
plt.figure()
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price per sqft")
plt.ylabel("Count")


print(df8.bath.unique())
print(df8[df8.bath > 10])

plt.figure()
plt.hist(df8.bath, rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")

# No of bathroom = no of bedrooms + 2
df9 = df8[df8.bath < df8.bhk + 2]
print(df9.shape)

df10 = df9.drop(['size', 'price_per_sqft'], axis=1)  # dropping unnecessary columns

# Model building
print(df10.head())
# location is a text column should be converted to numeric

dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop('other', axis=1)], axis=1)
df12 = df11.drop('location', axis=1)
print(df12.shape)

X = df12.drop('price', axis=1)
y = df12.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
reg = LinearRegression()
reg.fit(X_train.values, y_train)
print("Linear regression score ", reg.score(X_test, y_test))

# Model evaluation using cross-validation
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val = cross_val_score(LinearRegression(), X, y, cv=cv)
print("Linear regression with shufflesplit score ", cross_val)

# Finding the best model using GridSearchCV
def find_best_model(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    print("Starting grid search...")
    for algo_name, config in algos.items():
        try:
            print(f"Running GridSearchCV for {algo_name}...")
            gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
            gs.fit(X, y)
            print(f"GridSearchCV completed for {algo_name}")
            scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_,
                'cv_scores': gs.cv_results_['mean_test_score']
            })
        except Exception as e:
            print(f"An error occurred for {algo_name}: {str(e)}")

    print("Grid search completed.")
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params', 'cv_scores'])

# Rest of your code...

# Call the find_best_model function
best_model_info = find_best_model(X, y)
print("Best Model Information:")
print(best_model_info)


# Using linear_model
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]  # gives column index

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return reg.predict([x])[0]

predicted_value = predict_price('1st Phase JP Nagar', 1000, 2, 2)
print("Predicted value is ", predicted_value)

# Save the model
with open('house_price_model.pickle', 'wb') as f:
    pickle.dump(reg, f)

# Save column information
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))


plt.show()
