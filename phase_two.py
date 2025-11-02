import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Load dataset
df = pd.read_csv('ayurvedic_taste_dataset.csv')

# Separate features and labels
X = df[['pH', 'TDS', 'Var3']]
y = df[['Taste']]

# One-hot encode the taste labels
encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of deprecated sparse
y_encoded = encoder.fit_transform(y)

# Split into train (80%), validation (10%), test (10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)  # 0.1111 × 0.9 ≈ 0.1

# Normalize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'taste_scaler.pkl')

# Print dataset shapes and sample labels
print("✅ Data preparation complete.\n")
print("X_train shape:", X_train_scaled.shape)
print("X_val shape:", X_val_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("\nSample encoded labels:\n", y_encoded[:5])