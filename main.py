import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the training dataset into a DataFrame
train_df = pd.read_csv('emnist-digits-train.csv')

# Separate labels from features
X = train_df.iloc[:, 1:].values  # Features (pixel values)
y = train_df.iloc[:, 0].values    # Labels (digits)

# Normalize pixel values (scaling them between 0 and 1)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Reshape the feature data to represent images (28x28 pixels with 1 channel)
X_reshaped = X_normalized.reshape(-1, 28, 28, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Output layer with 10 units for 10 digits
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions on the test set
predictions = model.predict(X_test)

# Convert the predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Compare the predicted labels with the actual labels
correct_predictions = np.sum(predicted_labels == y_test)
total_samples = len(y_test)

print("Number of correctly predicted samples:", correct_predictions)
print("Total samples in the test set:", total_samples)
print("Accuracy on the test set:", correct_predictions / total_samples)
