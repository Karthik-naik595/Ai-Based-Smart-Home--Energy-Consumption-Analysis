import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ✅ Correct path to data file
file_path = os.path.join(os.path.dirname(__file__), '../data/cleaned_smart_home_data.csv')

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {os.path.abspath(file_path)}")

# ✅ Load data
data = pd.read_csv(file_path)

# ✅ Prepare data for prediction
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['energy_consumption_kwh']])

# Create sequences for simplicity, assuming this is more like a time-series prediction setup
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 10  # Can try reducing this if dataset is large
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# ✅ Reshape data for the Random Forest model (flatten the sequences for simplicity)
X = X.reshape((X.shape[0], X.shape[1]))  # Each row is a sequence of length 'SEQ_LENGTH'

# ✅ Split into train and test sets
split = int(0.8 * X.shape[0])
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ✅ Flatten y to be a 1D array (required for RandomForestRegressor)
y_train = y_train.ravel()  # Flatten y_train to be a 1D array
y_test = y_test.ravel()    # Flatten y_test to be a 1D array

# ✅ Build Random Forest model
model = RandomForestRegressor(n_estimators=10, random_state=42)  # Reduced n_estimators for testing

# ✅ Train the model
print("Training Random Forest model...")
try:
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
except Exception as e:
    print(f"❌ Error during training: {e}")

# ✅ Evaluate the model
try:
    y_pred = model.predict(X_test)
    print("✅ Prediction completed!")
except Exception as e:
    print(f"❌ Error during prediction: {e}")

# ✅ Rescale predictions
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))  # Reshape back to a column vector for inverse transform
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))    # Reshape to column vector for inverse transform

# ✅ Save Predictions
predictions_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
predictions_file = os.path.join(os.path.dirname(__file__), '../data/random_forest_predictions.csv')
predictions_df.to_csv(predictions_file, index=False)
print(f"✅ Predictions saved to: {os.path.abspath(predictions_file)}")

# ✅ Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ MAE: {mae:.4f}")

# ✅ Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')
plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Time Steps')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.show()
