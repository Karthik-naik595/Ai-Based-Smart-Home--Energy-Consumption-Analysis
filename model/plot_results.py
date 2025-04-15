import os
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load predictions
file_path = os.path.join(os.path.dirname(__file__), '../data/lstm_predictions.csv')
predictions_df = pd.read_csv(file_path)

# ✅ Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['Actual'], label='Actual', color='blue', alpha=0.7)
plt.plot(predictions_df['Predicted'], label='Predicted', color='red', alpha=0.7)
plt.title('LSTM - Actual vs Predicted Energy Consumption')
plt.xlabel('Time Step')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid()

# ✅ Save plot
plot_path = os.path.join(os.path.dirname(__file__), '../data/lstm_plot.png')
plt.savefig(plot_path)
print(f"✅ Plot saved to: {os.path.abspath(plot_path)}")

# ✅ Show plot
plt.show()
