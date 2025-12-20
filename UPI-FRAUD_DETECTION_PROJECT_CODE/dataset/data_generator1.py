import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for consistency
np.random.seed(42)

# 1. Generate a Large Pool of Data
# We generate 400,000 rows to ensure we have enough "Legit" cases (since your logic makes most cases Fraud)
num_rows_to_generate = 400000
print(f"Generating {num_rows_to_generate} rows to find balanced classes...")

trans_hour = np.random.randint(0, 24, num_rows_to_generate)
trans_day = np.random.randint(1, 29, num_rows_to_generate)
trans_month = np.random.randint(1, 13, num_rows_to_generate)
trans_year = np.random.randint(2017, 2026, num_rows_to_generate)
trans_amount = np.round(np.random.uniform(10, 200000, num_rows_to_generate), 2)
upi_number = np.random.randint(7662000000, 7662999999, num_rows_to_generate, dtype=np.int64)

# 2. Apply Strict Fraud Logic (Creates 'Perfect' Patterns)
fraud_risk_strict = (
    (trans_amount > 75000) |
    ((trans_hour < 6) & (trans_amount > 30000)) |
    ((trans_day > 25) & (trans_amount > 50000)) |
    ((trans_month % 2 == 0) & (trans_amount > 60000)) |
    ((trans_year >= 2023) & (trans_amount > 40000))
).astype(int)

# Create temporary DataFrame
data_large = pd.DataFrame({
    'trans_hour': trans_hour,
    'trans_day': trans_day,
    'trans_month': trans_month,
    'trans_year': trans_year,
    'trans_amount': trans_amount,
    'upi_number': upi_number,
    'fraud_risk': fraud_risk_strict
})

# 3. Balance the Classes (50k Legit, 50k Fraud)
df_legit = data_large[data_large['fraud_risk'] == 0]
df_fraud = data_large[data_large['fraud_risk'] == 1]

# Sample 50,000 from each (using replace=True to be safe, though likely not needed for Fraud)
df_legit_balanced = df_legit.sample(50000, replace=True, random_state=42)
df_fraud_balanced = df_fraud.sample(50000, replace=True, random_state=42)

data_balanced = pd.concat([df_legit_balanced, df_fraud_balanced])

# Shuffle the dataset
data_final = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Inject Noise (The Secret to 92% Accuracy)
# We flip 7% of the labels. This simulates "False Positives" and "False Negatives"
noise_percentage = 0.07 
mask = np.random.choice([True, False], size=len(data_final), p=[noise_percentage, 1-noise_percentage])

# Apply the flip: 0 becomes 1, 1 becomes 0 for the selected rows
data_final['fraud_risk'] = np.where(mask, 1 - data_final['fraud_risk'], data_final['fraud_risk'])

# Save final dataset
data_final.to_csv("upi_fraud_dataset_balanced.csv", index=False)

print("\n✅ Balanced & Realistic dataset generated!")
print("Final counts (approx 50/50):")
print(data_final['fraud_risk'].value_counts())

# --- Quick Test Verification ---
print("\nRunning quick test to verify accuracy...")
X = data_final.drop(columns=['fraud_risk'])
y = data_final['fraud_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Estimated Model Accuracy: {acc*100:.2f}%")