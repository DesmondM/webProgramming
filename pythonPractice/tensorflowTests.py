import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Training data: x values and corresponding y values
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Build a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model with loss function and optimizer
model.compile(optimizer='sgd', loss='mean_squared_error')

print("Training the model...")
# Train the model
history = model.fit(X, Y, epochs=500, verbose=False)
print("Model trained!")

# Test the model with a new value
test_value = 10.0
prediction = model.predict(np.array([[test_value]]))
print(f"Prediction for x={test_value}: {prediction[0][0]}")

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
