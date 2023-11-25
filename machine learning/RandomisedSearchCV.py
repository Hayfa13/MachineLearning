import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# Load the Iris dataset, which is a popular dataset for classification problems
iris = load_iris()

rs=RandomizedSearchCV(SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['linear','rbf']
},cv=5,return_train_score=False,n_iter=2)
rs.fit(iris.data,iris.target)
df = pd.DataFrame(rs.cv_results_)
print(df[['param_C','param_kernel','mean_test_score']])
