import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

digits = load_digits()  # You need to call the function to load the dataset
print(dir(digits))
print(digits.data[0])  # Printing the first data point

print(digits.target[0:5])
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.8)
reg = LogisticRegression()
reg.fit(X_train, y_train)
score = reg.score(X_test, y_test)
print("The accuracy is", score)

predict = reg.predict(digits.data[[67]])
print("The predicted value for sample is", predict)

y_predicted = reg.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
print(cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel("Truth")
plt.show()
