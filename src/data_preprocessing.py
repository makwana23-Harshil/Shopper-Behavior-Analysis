import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Handle missing values
    df.fillna({
        'Age': df['Age'].median(),
        'Purchase Amount (USD)': df['Purchase Amount (USD)'].median(),
        'Review Rating': df['Review Rating'].median()
    }, inplace=True)

    # Encode categorical columns
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
        df[col] = encoder.fit_transform(df[col])

    # Scale numerical columns
    scaler = StandardScaler()
    numeric_cols = [
        'Age',
        'Purchase Amount (USD)',
        'Previous Purchases',
        'Frequency of Purchases',
        'Review Rating'
    ]

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save processed data
    df.to_csv(output_path, index=False)
    print("✅ Data preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw_data.csv",
        output_path="data/processed_data.csv"
    )

