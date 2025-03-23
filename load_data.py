import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
df = pd.read_csv("EcommerceDataset.csv")
print("\nAvailable columns in the dataset:")
print(list(df.columns))
#print(df.columns)  # Check real column names
df.dropna(inplace=True)  # Or use fillna if you prefer
df['HourSpendOnApp'].fillna(df['HourSpendOnApp'].mean(), inplace=True)
df['OrderCount'].fillna(0, inplace=True)

# Show basic info
print("\nFirst 5 rows:")
print(df.head())
print("\nShape of dataset:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Check column names and types
print("\nColumn names:")
print(df.columns)
print("\nData types:")
print(df.dtypes)

# Look at churn value distribution
print("\nChurn value counts:")
print(df['Churn'].value_counts())

# Drop non-informative ID column
df = df.drop(columns=['CustomerID'])

# Convert categorical columns to dummy variables
categorical_columns = ['Gender', 'MaritalStatus', 'PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Confirm all columns are now numeric
print("\nUpdated DataFrame info:")
print(df.info())

# Separate features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
model = LogisticRegression(max_iter=1000)  # max_iter ensures convergence

# Train the model
model.fit(X_train, y_train)

# Predict churn on test data
y_pred = model.predict(X_test)

# Evaluate performance
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_y_pred = rf_model.predict(X_test)

# Evaluate performance
print("\n🌲 Random Forest Model Accuracy:", accuracy_score(y_test, rf_y_pred))
print("\n📊 Random Forest Classification Report:\n", classification_report(y_test, rf_y_pred))
print("\n🧮 Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_y_pred))

import joblib
joblib.dump(rf_model, "churn_model.pkl")
joblib.dump(list(X.columns), 'model_features.pkl')




