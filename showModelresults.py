from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i].reshape(28, 28)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label, 100 * np.max(predictions_array), true_label), color=color)

def show_results():
    predictions = model.predict(test_images)
    num_rows = 6
    num_cols = 12
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plot_image(i, predictions, test_labels, test_images)
    plt.show()


if __name__ == "__main__":
    file_name = input("file_name(path) :")
    model = keras.models.load_model(file_name.strip())
    model.evaluate(test_images, test_labels)
    show_results()
