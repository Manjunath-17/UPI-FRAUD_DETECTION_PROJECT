
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\manjunath s khot\Documents\2KD22CS049\UPI-FRAUD_DETECTION_PROJECT_CODE_final\cleaned_upi_fraud_dataset.csv")

# Convert 'upi_number' to string (in case it’s numeric)
data["upi_number"] = data["upi_number"].astype(str)

# Encode UPI numbers (convert text → numeric)
label_encoder = LabelEncoder()
data["upi_number"] = label_encoder.fit_transform(data["upi_number"])

# Define features and target
X = data[["trans_hour", "trans_day", "trans_month", "trans_year", "trans_amount", "upi_number"]]
y = data["fraud_risk"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save model, scaler, and encoder
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model, Scaler, and Encoder saved successfully!")
