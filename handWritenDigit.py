from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# mnist : base de données d'image labélisé de chiffres manuscrite de taille 28x28
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)

#extraction du set de validation
val_images = train_images[50000:]
train_images = train_images[:50000]
val_labels = train_labels[50000:]
train_labels = train_labels[:50000]

# construction du modèle
# la couche d'entrée applique une convolution avec une matrice de taille 5x5 et utilise la fonction d'activation relu
# sortie : 28x28x4
# les 2 et 3ième couches appliquent auusi des convolutions tout en divisant par 4 la dimension de leurs sorties
# sorties : 14x14x7 et 7x7x12
# la 4ème redimensionne son entrée en un vecteur unidimentionnel
# sortie : 588x1
# la 5ème couche est calcule les prédictions
# sortie : 10x1
# la couche de sortie calcule les prédictions sous formed'une distribution de probabilité avec la fonction softmax
# sortie : 10x1

model = keras.Sequential([
    tf.layers.Conv2D(4, 5, 1, padding="same", use_bias=True, activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.layers.Conv2D(7, 4, 2, padding="same", use_bias=True, activation=tf.nn.relu),
    tf.layers.Conv2D(12, 4, 2, padding="same", use_bias=True, activation=tf.nn.relu),
    tf.layers.Flatten(),
    tf.layers.Dense(200, activation=tf.nn.relu),
    tf.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
          batch_size=100,
          epochs=12,
          verbose=1,
          validation_data=(val_images, val_labels))

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

tf.keras.models.save_model(
    model,
    filepath="handWrittenDigitModel",
    overwrite=True,
    include_optimizer=True
)

history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()