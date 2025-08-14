# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
data = pd.read_csv('creditcard.csv')

# Step 2: Data exploration
print(data.head())
print(data.info())
print(data.describe())
                                    
# Step 3: Checking for null values
print(data.isnull().sum())

# Step 4: Data Preprocessing
# The dataset is highly imbalanced, so it's important to handle that.
fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0]

# Downsample the non-fraud cases
non_fraud_sample = non_fraud.sample(n=len(fraud))

# Create a balanced dataset
balanced_data = pd.concat([fraud, non_fraud_sample])

# Split into features and target
X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Model Building
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = lr_model.predict(X_test)

# Accuracy and classification report
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()

# Step 9: Random Forest for comparison
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Random Forest results
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion matrix visualization for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d')
plt.show()
