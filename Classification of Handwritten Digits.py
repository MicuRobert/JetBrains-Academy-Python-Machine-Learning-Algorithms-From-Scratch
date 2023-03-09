import numpy as np
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)

print(f'Classes: {np.unique(y_train)}')
print(f'Features\' shape: {np.shape(x_train)}')
print(f'Target\'s shape: {np.shape(y_train)}')
print(f'min: {np.min(x_train)}, max: {np.max(x_train)}')
