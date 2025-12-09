import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained model
model = load_model('model_file_30epochs.h5')

# Set the path to your test data
test_data_dir = '/Users/harishwar/PycharmProjects/facialemotionrecognition/fer2013/test'

# Define the data generator for test data
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create the test data generator
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    color_mode='grayscale',  # Adjust based on your model input
    target_size=(48, 48),    # Adjust based on your model input
    batch_size=32,
    class_mode='categorical',  # Adjust based on your model output
    shuffle=False  # Do not shuffle for evaluation
)

# Get the number of test samples
num_test_samples = len(test_generator.filenames)

# Evaluate the model on the test set
evaluation = model.evaluate(test_generator, steps=num_test_samples // 32)

# Extract and print the accuracy
accuracy = evaluation[1]
print(f'Model Accuracy on Test Set: {accuracy * 100:.2f}%')
