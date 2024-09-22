import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv(r'C:\Users\daher\OneDrive\Desktop\University Materials\Semester 6\AI Engineering\Week 4\vegemite.csv')

# Display the first few rows and info
print("First few rows of the dataset:")
print(data.head())
print("\nDataset info:")
print(data.info())

# Preprocessing
class_column_name = 'Class'
features = data.columns[data.columns != class_column_name]
X = data[features]
y = data[class_column_name]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(random_state=42)

# Fit models
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Predictions
dt_predictions = decision_tree.predict(X_test)
rf_predictions = random_forest.predict(X_test)
lr_predictions = logistic_regression.predict(X_test)
svm_predictions = svm.predict(X_test)

# Classification Reports
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print("\nSupport Vector Machine Classification Report:")
print(classification_report(y_test, svm_predictions))
print("Support Vector Machine Confusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))

