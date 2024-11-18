import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate synthetic data for linear regression
# y = 2x + 1 + noise
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.5

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.show()

# Create a linear regression model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model with mean squared error loss and an optimizer
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model on the data
model.fit(X, y, epochs=100)

# Make predictions
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

print("Predictions:", y_predict)
