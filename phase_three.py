import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv('ayurvedic_taste_dataset.csv')
X = df[['pH', 'TDS', 'Var3']]
y = df[['Taste']]

# One-hot encode labels
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Save encoder for deployment
joblib.dump(encoder, 'taste_label_encoder.pkl')

# Split and scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'taste_scaler.pkl')

# Build MLP model
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dropout(0.2),
    Dense(12, activation='relu'),
    Dense(6, activation='softmax')  # 6 taste classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
                    epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Save model
model.save('taste_mlp_model.h5')