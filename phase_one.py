import numpy as np
import pandas as pd

# Define Ayurvedic taste ranges
taste_ranges = {
    'Madhura':  {'pH': (6.3, 7.2), 'TDS': (100, 500)},
    'Amla':     {'pH': (2.0, 5.0), 'TDS': (200, 600)},
    'Lavana':   {'pH': (6.8, 8.0), 'TDS': (500, 1500)},
    'Katu':     {'pH': (5.5, 7.5), 'TDS': (250, 600)},
    'Tikta':    {'pH': (7.0, 8.9), 'TDS': (350, 700)},
    'Kashaya':  {'pH': (5.0, 6.8), 'TDS': (400, 800)}
}

samples_per_taste = 2000
data = []

# Generate samples
for taste, ranges in taste_ranges.items():
    pH = np.random.uniform(*ranges['pH'], samples_per_taste)
    TDS = np.random.uniform(*ranges['TDS'], samples_per_taste)
    Var3 = np.random.uniform(0.1, 1.0, samples_per_taste)  # Optional third variable
    label = [taste] * samples_per_taste
    df = pd.DataFrame({'pH': pH, 'TDS': TDS, 'Var3': Var3, 'Taste': label})
    data.append(df)

# Combine and shuffle
full_data = pd.concat(data).sample(frac=1).reset_index(drop=True)

# Export to CSV
full_data.to_csv('ayurvedic_taste_dataset.csv', index=False)

# Quick verification
print("Summary Statistics:")
print(full_data.describe())
print("\nTaste Class Counts:")
print(full_data['Taste'].value_counts())