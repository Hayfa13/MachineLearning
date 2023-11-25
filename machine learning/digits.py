import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()

print("Description of dataset ",dir(digits))
df=pd.DataFrame(digits.data,digits.target)
print()
print(df.head())
df['target']=digits.target
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)
reg=SVC(kernel='linear')
reg.fit(X_train,y_train)
score=reg.score(X_test,y_test)
print("The score is ", score)