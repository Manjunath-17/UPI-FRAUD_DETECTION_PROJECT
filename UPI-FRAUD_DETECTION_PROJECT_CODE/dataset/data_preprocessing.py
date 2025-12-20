import pandas as pd

def clean_upi_data(file_path, output_path):
    """
    Cleans the UPI fraud dataset by removing duplicates and validating logical ranges.
    """
    # 1. Load the dataset
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Original shape: {df.shape}")
    except FileNotFoundError:
        print("File not found. Please check the path.")
        return

    # 2. Remove Duplicate Rows
    # This keeps the first occurrence and removes subsequent identical rows
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    duplicates_removed = initial_count - len(df_cleaned)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows.")
    else:
        print("No duplicate rows found.")

    # 3. Validate Logical Ranges
    # Ensure time and date columns contain valid values
    # hour: 0-23, day: 1-31, month: 1-12
    valid_condition = (
        (df_cleaned['trans_hour'].between(0, 23)) &
        (df_cleaned['trans_day'].between(1, 31)) &
        (df_cleaned['trans_month'].between(1, 12))
    )
    
    # Filter data to keep only valid rows
    invalid_rows = (~valid_condition).sum()
    df_cleaned = df_cleaned[valid_condition]
    
    if invalid_rows > 0:
        print(f"Removed {invalid_rows} rows with invalid date/time values.")

    # 4. Save the cleaned data
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaning complete. Saved to '{output_path}'. Final shape: {df_cleaned.shape}")

# Execute the cleaning function
clean_upi_data('upi_fraud_dataset_balanced.csv', 'cleaned_upi_fraud_dataset.csv')