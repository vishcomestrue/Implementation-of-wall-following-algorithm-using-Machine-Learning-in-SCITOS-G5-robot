from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


data=pd.read_csv('data.csv')
data.head()

X=data.iloc[:, :-1]

Y=data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = tree.DecisionTreeClassifier(random_state=42)

clf = clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy of Decision Tree model: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True)
plt.show()
