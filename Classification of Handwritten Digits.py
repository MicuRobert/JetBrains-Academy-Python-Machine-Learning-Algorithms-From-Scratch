import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)

x_train = x_train[:6000]
y_train = y_train[:6000]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=40)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

train_class_proportions = pd.Series(y_train).value_counts(normalize=True)
print("Proportion of samples per class in train set:\n", train_class_proportions)
