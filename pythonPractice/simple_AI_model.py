import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Training data
house_sizes = np.array([50, 60, 80, 100, 120, 150], dtype=int)
house_prices = np.array([100, 120, 160, 200, 240, 300], dtype=int)

# 2. Normalize the data (better for training)
house_sizes_norm = house_sizes / 150.0    # max size = 150
house_prices_norm = house_prices / 300.0  # max price = 300

# 3. Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train the model
model.fit(house_sizes_norm, house_prices_norm, epochs=1000, verbose=False)

# 5. Make a prediction for a 90 m² house
predicted_price = model.predict(np.array([90 / 150])) * 300
print(f"Predicted price for 90 m² house: ${predicted_price[0][0]:.2f}k")

# 6. Plot the data and prediction line
plt.scatter(house_sizes, house_prices, color='blue', label='Training Data')

# Generate predicted line for visualization
test_sizes = np.linspace(40, 160, 100)        # 40 to 160 m²
predicted_line = model.predict(test_sizes / 150.0) * 300

plt.plot(test_sizes, predicted_line, color='red', label='Model Prediction')
plt.xlabel("House Size (m²)")
plt.ylabel("Price ($1000s)")
plt.title("House Price Prediction")
plt.legend()
plt.show()
