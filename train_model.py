import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt




# Load dataset
df = pd.read_csv("EcommerceDataset.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Drop ID column if present
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Find all non-numeric (categorical) columns automatically
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Print for debugging
print("Encoding categorical columns:", categorical_columns)

# One-hot encode them
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)


# Separate features and target
print("\n🔍 Columns before training:")
print(df.dtypes)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Balanced churn classes:\n", pd.Series(y_resampled).value_counts())


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and expected feature names
joblib.dump(model, "churn_model.pkl")
joblib.dump(list(X.columns), "model_features.pkl")
importances = model.feature_importances_
feature_names = X.columns

# Plot
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

print(feat_df.head(10))
feat_df.plot(kind='barh', x='Feature', y='Importance', figsize=(10, 6))
plt.title("Top Features Influencing Churn")
plt.tight_layout()
plt.show()
print("\n🔍 Top 10 Features Influencing Churn:")
print(feat_df.head(10))
print("✅ Model trained and saved successfully!")
