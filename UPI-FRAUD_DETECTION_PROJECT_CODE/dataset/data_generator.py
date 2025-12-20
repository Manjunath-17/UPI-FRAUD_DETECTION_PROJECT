import pandas as pd
import numpy as np

# Set random seed for consistency
np.random.seed(42)

num_rows = 100000

# Generate features
trans_hour = np.random.randint(0, 24, num_rows)
trans_day = np.random.randint(1, 29, num_rows)
trans_month = np.random.randint(1, 13, num_rows)
trans_year = np.random.randint(2017, 2026, num_rows)
trans_amount = np.round(np.random.uniform(10, 200000, num_rows), 2)
upi_number = np.random.randint(7662000000, 7662999999, num_rows, dtype=np.int64)

# Improved fraud logic
fraud_risk = (
    (trans_amount > 75000) |
    ((trans_hour < 6) & (trans_amount > 30000)) |
    ((trans_day > 25) & (trans_amount > 50000)) |
    ((trans_month % 2 == 0) & (trans_amount > 60000)) |
    ((trans_year >= 2023) & (trans_amount > 40000))
).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'trans_hour': trans_hour,
    'trans_day': trans_day,
    'trans_month': trans_month,
    'trans_year': trans_year,
    'trans_amount': trans_amount,
    'upi_number': upi_number,
    'fraud_risk': fraud_risk
})

# Save dataset
data.to_csv("upi_fraud_dataset.csv", index=False)
print("✅ Clean dataset generated and saved as 'upi_fraud_dataset.csv'")
