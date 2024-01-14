from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]
print(train_labels[4])
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
