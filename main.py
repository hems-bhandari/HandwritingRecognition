import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from PIL import Image
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the model to a file
model.save('mnist_model.h5')

# Load the model from the saved file
model = tf.keras.models.load_model('mnist_model.h5')

# Load a new image
image = Image.open('sample_image_2.png')
image = image.convert('L')
image = image.resize((28, 28))
image_array = np.array(image)
image_array = image_array / 255.0
image_array = np.reshape(image_array, (1, 28, 28))

# Make a prediction on the new image
prediction = model.predict(image_array)
digit = np.argmax(prediction)

# Display the predicted digit
print('The predicted digit is:', digit)