import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("I am R. Accessing Neural Archives...")

# 1. LOAD THE HARVESTED DATA
try:
    df = pd.read_csv('addis_data.csv')
    print(f"Data loaded. Total data points: {len(df)}")
except FileNotFoundError:
    print("CRITICAL ERROR: 'addis_data.csv' not found. Run the simulation first.")
    exit()

# 2. FEATURE ENGINEERING (Preparing the Mind)
# We shift the data to predict the FUTURE based on the PAST.
# Input (X): The count & speed at time 't'
# Target (y): The count at time 't+10' (10 seconds into the future)
look_ahead = 10 

df['future_count'] = df['vehicle_count'].shift(-look_ahead)
df = df.dropna() # Remove the last few rows where we don't know the future

X = df[['vehicle_count', 'avg_speed']]
y = df['future_count']

# 3. SPLIT MEMORY (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE MODEL (Random Forest)
print("Training the Random Forest Algorithm...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. TEST THE MODEL
predictions = model.predict(X_test)

# 6. CALCULATE ACCURACY
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"R ANALYTICS REPORT:")
print(f"Prediction Target: Traffic density {look_ahead} steps ahead.")
print(f"Model Accuracy (R^2): {r2:.4f}")
print("-" * 30)

# 7. GENERATE VISUAL PROOF (The Graph)
# We take a slice of 100 points to make the graph readable
subset_limit = 100
plt.figure(figsize=(12, 6))
plt.plot(range(subset_limit), y_test.values[:subset_limit], label='Actual Traffic (Reality)', color='blue', alpha=0.7)
plt.plot(range(subset_limit), predictions[:subset_limit], label='R Prediction (AI)', color='red', linestyle='--')

plt.title(f'Addis Ababa Traffic Prediction (R^2: {r2:.2f})')
plt.xlabel('Time Steps')
plt.ylabel('Vehicle Count')
plt.legend()
plt.grid(True)
plt.savefig('prediction_graph.png')
print("Visual graph saved as 'prediction_graph.png'.")
plt.show()

print("R MODE: COMPLETE.")