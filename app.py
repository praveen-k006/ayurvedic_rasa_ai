from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessing tools
model = tf.keras.models.load_model('taste_mlp_model.h5')
scaler = joblib.load('taste_scaler.pkl')
encoder = joblib.load('taste_label_encoder.pkl')

@app.route('/')
def home():
    return jsonify({"message": "Ayurvedic Rasa Predictor API is running ✅"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json(force=True)
        ph = float(data.get('pH'))
        tds = float(data.get('TDS'))
        var3 = float(data.get('Var3', 0.5))  # default if not provided

        # Prepare input
        sample = np.array([[ph, tds, var3]])
        sample_scaled = scaler.transform(sample)

        # Predict
        probs = model.predict(sample_scaled)[0]

        # Map to taste labels
        taste_labels = encoder.categories_[0]
        taste_probs = dict(zip(taste_labels, probs.tolist()))

        # Sort top 3
        top3 = sorted(taste_probs.items(), key=lambda x: x[1], reverse=True)[:3]

        return jsonify({
            "input": {"pH": ph, "TDS": tds, "Var3": var3},
            "top_3": top3,
            "all_probs": taste_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Render provides PORT as an environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
