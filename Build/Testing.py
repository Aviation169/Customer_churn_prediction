import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("best_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load feature names from training data
feature_names = joblib.load("feature_names.pkl")  # Save this during training

# Example data
new_data_dict = {
    'gender_Male': [1], 'SeniorCitizen_Yes': [1], 'Partner_Yes': [1], 'Dependents_Yes': [1],
    'tenure': [5], 'PhoneService_Yes': [1], 'MultipleLines_Yes': [0], 'InternetService_Fiber optic': [0],
    'OnlineSecurity_Yes': [0], 'OnlineBackup_Yes': [0], 'DeviceProtection_Yes': [1], 'TechSupport_Yes': [0],
    'StreamingTV_Yes': [0], 'StreamingMovies_Yes': [1], 'Contract_One year': [0], 'Contract_Two year': [1],
    'PaperlessBilling_Yes': [0], 'PaymentMethod_Credit card (automatic)': [1], 'PaymentMethod_Electronic check': [0],
    'PaymentMethod_Mailed check': [0], 'MonthlyCharges': [330], 'TotalCharges': [6600]
}

# Convert to DataFrame
new_data_df = pd.DataFrame(new_data_dict)

# Ensure columns match the trained model
new_data_df = new_data_df.reindex(columns=feature_names, fill_value=0)  # Add missing columns if needed

# Scale the data
new_data_scaled = scaler.transform(new_data_df)

# Predict churn probability
churn_probability = model.predict_proba(new_data_scaled)[:, 1]
churn_prediction = model.predict(new_data_scaled)

print(f"🔹 Churn Probability: {churn_probability[0]:.4f}")
print(f"🔹 Churn Prediction: {'Yes' if churn_prediction[0] == 1 else 'No'}")