import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])  # Matching target replaced with target names

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
reg = SVC(kernel='rbf', C=30, gamma='auto')
reg.fit(X_train, y_train)  # Fixed: fitting on the training data, not on X_test and y_test

kernel = ['linear', 'rbf']
C = [1, 10, 20]
avg_scores = {}

for kval in kernel:
    for cval in C:
        clf = SVC(kernel=kval, C=cval, gamma='auto')
        cv_scores = cross_val_score(clf, iris.data, iris.target, cv=5)
        avg_scores[kval + '_' + str(cval)] = np.average(cv_scores)

for key, value in avg_scores.items():
    print(key, ":", value)



