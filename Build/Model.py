import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Load Data
df = pd.read_csv("/content/ML_project1.csv")  

# Drop Unnecessary Column
df.drop(columns=['customerID'], inplace=True)

# Convert 'TotalCharges' to Numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values for numeric features only
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Convert 'SeniorCitizen' to Categorical
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# Convert Target Variable 'Churn' to Binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding for Categorical Variables
df = pd.get_dummies(df, drop_first=True)

# Split Data into Features and Target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# Train Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and Evaluate Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"🔹 Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 60)

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model after Tuning
best_model = grid_search.best_estimator_
print("🔹 Best Parameters:", grid_search.best_params_)

# Save the Best Model
joblib.dump(best_model, "best_churn_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Feature Importance Visualization
feature_importance = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Important Features in Churn Prediction")
plt.show()
# Final Model Evaluation
y_pred_best = best_model.predict(X_test)
print("🔹 Final Model Evaluation")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]):.4f}")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))