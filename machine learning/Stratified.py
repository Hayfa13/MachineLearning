import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
digits=load_digits()
df=pd.DataFrame(digits.data,digits.target)
df['target']=digits.target
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)
kf=KFold(n_splits=3)


def get_score(reg,X_train, X_test, y_train, y_test):
    reg.fit(X_train,y_train)
    return reg.score(X_test,y_test)

print(cross_val_score(SVC(),digits.data,digits.target))
