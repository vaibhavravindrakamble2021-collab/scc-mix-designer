import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the Excel data
df = pd.read_excel("dataset.SCC.xlsx", sheet_name="Sheet1")

# 2. Separate the Input (Strength) and Outputs (Material Quantities)
# We want to input the Strength (Y) to get the Mix Proportions (D0-D5)
X = df[['Y']] 
y = df[['D0', 'D1', 'D2', 'D3', 'D4', 'D5']] 

# 3. Scale the input data so the neural network learns efficiently
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Build and train the Artificial Neural Network
# Using 2 hidden layers with 100 and 50 neurons respectively
print("Training the ANN model... Please wait.")
ann_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
ann_model.fit(X_scaled, y)

# 5. Save the trained model and the scaler to your computer
joblib.dump(ann_model, "scc_model.pkl")
joblib.dump(scaler, "scc_scaler.pkl")

print("Training complete! Model saved as 'scc_model.pkl'.")