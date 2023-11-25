import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
digits=load_digits()
df=pd.DataFrame(digits.data,digits.target)
df['target']=digits.target
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)
kf=KFold(n_splits=3)

for train_index ,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index,test_index)

def get_score(reg,X_train, X_test, y_train, y_test):
    reg.fit(X_train,y_train)
    return reg.score(X_test,y_test)

print(get_score(SVC(),X_train, X_test, y_train, y_test))