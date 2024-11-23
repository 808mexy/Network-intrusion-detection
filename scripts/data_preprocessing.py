# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset (Make sure to replace the path with the correct file location on your computer)
df = pd.read_csv('C:/Users/19562/Downloads/Thursday-WorkingHours-Morning-WebAttacks.csv')

# Drop unnecessary columns
columns_to_drop = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', ' Source Port', ' Destination Port']
df_cleaned = df.drop(columns=columns_to_drop)

# Handle missing values
df_cleaned.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df_cleaned = df_cleaned.dropna()

# Encode labels
label_encoder = LabelEncoder()
df_cleaned[' Label'] = label_encoder.fit_transform(df_cleaned[' Label'])

# Prepare features and labels
features = df_cleaned.drop(columns=[' Label'])
labels = df_cleaned[' Label']

# Feature scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels, test_size=0.3, random_state=42
)
