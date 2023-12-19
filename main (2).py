import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Assume you have a dataset with images and labels
# Replace this with your actual data loading and preprocessing code
# For simplicity, let's assume 'X' contains images and 'y' contains labels

# Example data loading (replace this with your actual data loading code)
X = np.random.random((1000, 64, 64, 3))  # Random images for illustration
y = np.random.randint(2, size=(1000,))

# Preprocess the data (replace this with your actual preprocessing code)
X = X / 255.0  # Normalize pixel values to [0, 1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Make predictions on new data
new_data = np.random.random((5, 64, 64, 3))  # Replace with your actual new data
predictions = model.predict(new_data)
print("Predictions:", predictions)
