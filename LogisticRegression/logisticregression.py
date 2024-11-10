import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = '/content/data_classified.csv'
data = pd.read_csv(file_path)

label_encoder = LabelEncoder()
data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


clfs = LogisticRegression(random_state=0, max_iter=1000, C=10, solver='lbfgs')
clfs.fit(X_train, y_train)


y_train_pred = clfs.predict(X_train)
y_test_pred = clfs.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Precision, Recall, and F1-Score
train_precision = precision_score(y_train, y_train_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')

train_recall = recall_score(y_train, y_train_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Output
print("Training Accuracy:", train_acc * 100)
print("Testing Accuracy:", test_acc * 100)

print("\nTraining Precision:", train_precision)
print("Testing Precision:", test_precision)

print("\nTraining Recall:", train_recall)
print("Testing Recall:", test_recall)

print("\nTraining F1-Score:", train_f1)
print("Testing F1-Score:", test_f1)
