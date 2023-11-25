import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
digits = load_digits()

print("Description of dataset ",dir(digits))
df=pd.DataFrame(digits.data,digits.target)
df['target']=digits.target
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3)

reg=RandomForestClassifier(n_estimators=30)
reg.fit(X_train,y_train)
score=reg.score(X_test,y_test)
print("The score is ",score)

y_predicted =reg.predict(X_test)

cm= confusion_matrix(y_test,y_predicted)
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()