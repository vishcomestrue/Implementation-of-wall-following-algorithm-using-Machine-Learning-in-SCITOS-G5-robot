import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
file_path = '/content/data_classified.csv'
data = pd.read_csv(file_path)

# Encode the target column
label_encoder = LabelEncoder()
data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])

# Separate features and target
X = data.iloc[:, :-1]  # all columns except the last one as features
y = data.iloc[:, -1]   # last column as target
X1 = X.iloc[:, :2]

# Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=23)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=23)



svm_clf = SVC(kernel='rbf',random_state=0)
svm_clf.fit(X_train, y_train)



# Predictions
y_train_pred = svm_clf.predict(X_train)
y_val_pred = svm_clf.predict(X_val)
y_test_pred = svm_clf.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Precision, Recall, and F1-Score for each set
train_precision = precision_score(y_train, y_train_pred, average='weighted')
val_precision = precision_score(y_val, y_val_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')

train_recall = recall_score(y_train, y_train_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Output results
print("Training Accuracy:", train_acc * 100)
print("Validation Accuracy:", val_acc * 100)
print("Testing Accuracy:", test_acc * 100)

print("\nTraining Precision:", train_precision)
print("Validation Precision:", val_precision)
print("Testing Precision:", test_precision)

print("\nTraining Recall:", train_recall)
print("Validation Recall:", val_recall)
print("Testing Recall:", test_recall)

print("\nTraining F1-Score:", train_f1)
print("Validation F1-Score:", val_f1)
print("Testing F1-Score:", test_f1)
