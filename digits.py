from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import imageio
import sys


images = []
labels = []

for filename in os.listdir("coil-20-proc"):
    filepath = "coil-20-proc/" + filename
    im = imageio.imread(filepath)
    images.append(im)
    if filename[4] == '_':
        labels.append(int(filename[3])-1)
    else:
        labels.append(int(filename[3:5])-1)

class_names = []
for i in range(20):
    class_names.append("obj{0}".format(i))

print(class_names)

test_images = []
test_labels = []
train_images = []
train_labels = []
# Splice into training sets and test sets
for i in range(len(images)):
    if i % 10 == 5:
        test_images.append(images[i])
        test_labels.append(labels[i])
    else:
        train_images.append(images[i])
        train_labels.append(labels[i])


'''
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''


def divideImageArray(imageArr):
    for i in range(len(imageArr)):
        for rowIdx in range(len(imageArr[i])):
            for colIdx in range(len(imageArr[i][rowIdx])):
                imageArr[i][rowIdx][colIdx] = imageArr[i][rowIdx][colIdx] / 255.0

test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)                

test_images = test_images / 255.0
train_images = train_images / 255.0



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128, 128)), # Size of image is 128 x 128
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(20, activation='softmax') # Number of classifiers
])

'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(20))
    plt.yticks([])
    thisplot = plt.bar(range(20), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

