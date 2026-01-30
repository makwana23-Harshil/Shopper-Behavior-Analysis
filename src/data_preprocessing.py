import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Numeric columns
    numeric_cols = [
        'Age',
        'Purchase Amount (USD)',
        'Previous Purchases',
        'Frequency of Purchases',
        'Review Rating'
    ]

    # Convert to numeric safely
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing numeric values
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    categorical_cols = [
        'Gender',
        'Category',
        'Payment Method',
        'Shipping Type',
        'Discount Applied',
        'Promo Code Used',
        'Subscription Status',
        'Preferred Payment Method',
        'Season'
    ]

    encoder = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = encoder.fit_transform(df[col].astype(str))

    # Scale numeric data
    scaler = StandardScaler()
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    df[existing_numeric_cols] = scaler.fit_transform(df[existing_numeric_cols])

    # Save cleaned data
    df.to_csv(output_path, index=False)

    return df
