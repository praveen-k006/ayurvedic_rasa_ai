import tensorflow as tf
import numpy as np
import joblib

# Load model and preprocessing tools
model = tf.keras.models.load_model('taste_mlp_model.h5')
scaler = joblib.load('taste_scaler.pkl')
encoder = joblib.load('taste_label_encoder.pkl')

# Input sample
sample = np.array([[6.5, 450, 0.6]])  # pH, TDS, Var3

# Scale input
sample_scaled = scaler.transform(sample)

# Predict probabilities
probs = model.predict(sample_scaled)[0]

# Map to taste labels
taste_labels = encoder.categories_[0]
taste_probs = dict(zip(taste_labels, probs))

# Sort and display top 3
top_3 = sorted(taste_probs.items(), key=lambda x: x[1], reverse=True)[:3]

print("✅ Top 3 Predicted Tastes for Sample (pH=5.5, TDS=550, Var3=0.6):\n")
for taste, prob in top_3:
    print(f"Taste: {taste} — {prob*100:.2f}%")
