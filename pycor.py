import time
import numpy as np
import tensorflow as tf
# import tflite_runtime.interpreter as tf
import sys
sys.path.append('/Users/karimasadykova/Downloads/pycoral/pycoral')
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter

# Load the model onto the EdgeTPU
model_path = 'test_data/tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Divide the dataset into train and test data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data to match the input format expected by ResNet-50
def preprocess(image):
    # Resize the image to 224x224 pixels
    image = tf.image.resize(image, [224, 224])
    # Convert the image to a float32 tensor
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values to the range [-1, 1]
    image = (image / 127.5) - 1.0
    # Expand the dimensions to create a batch of size 1
    image = tf.expand_dims(image, 0)
    return image

test_data = [preprocess(image) for image in test_images]

# Save the train and test data to separate files or in memory
# Save the train and test data to separate files
np.save('test_data.npy', test_data)

# Load the train and test data from files
# test_data = np.load('test_data.npy')

# Loop through the test data in batches of the desired size, and feed the data to the model
batch_size = 32
total_time = 0.0
num_batches = len(test_data) // batch_size

for i in range(num_batches):
    batch = test_data[i*batch_size:(i+1)*batch_size]
    inputs = np.array(batch)
    start_time = time.time()
    outputs = classify.get_classes(interpreter)
    end_time = time.time()
    print(f"Time for batch {i}: {end_time - start_time}")
    total_time += end_time - start_time

avg_time_per_batch = total_time / num_batches
total_time = avg_time_per_batch * (len(test_data) // batch_size)

print(f"Total time to process test data with batch size {batch_size}: {total_time} seconds")

