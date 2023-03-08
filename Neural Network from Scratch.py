import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt


def scale(X):
    X = X / np.max(X)
    return X

def xavier(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.array(np.random.uniform(-limit, limit, (n_in, n_out)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse( y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train

def train(model, X, y, alpha, batch_size=100):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        model.backprop(X[i:i + batch_size], y[i:i + batch_size], alpha)

def accuracy(model, X, y):
    y_pred = np.argmax(model.forward(X), axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true)

class OneLayerNeural:
    def __init__(self, n_features, n_classes):
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)

    def forward(self, X):
        return sigmoid(np.dot(X, self.weights) + self.biases)

    def backprop(self, X, y, alpha):
        error = (mse_derivative(self.forward(X), y) * sigmoid_derivative(np.dot(X, self.weights) + self.biases))

        delta_W = (np.dot(X.T, error)) / X.shape[0]
        delta_b = np.mean(error, axis=0)

        self.weights -= alpha * delta_W
        self.biases -= alpha * delta_b

class TwoLayerNeural:
    def __init__(self, n_features, n_classes, hidden_size = 64):
        self.weights = [xavier(n_features, hidden_size), xavier(hidden_size, n_classes)]
        self.biases = [xavier(1, hidden_size), xavier(1, n_classes)]

    def forward(self, X):
        for i in range(2):
            X = sigmoid(np.dot(X, self.weights[i]) + self.biases[i])
        return X

    def backprop(self, X, y, alpha):
        n = X.shape[0]
        vec_ones = np.ones((1, n))
        yp = self.forward(X)
        db1 = 2 * alpha / n * ((yp - y) * yp * (1 - yp))
        f1 = sigmoid(np.dot(X, self.weights[0]) + self.biases[0])
        db0 = (np.dot(db1, self.weights[1].T)) * f1 * (1 - f1)

        self.weights[0] -= np.dot(X.T, db0)
        self.weights[1] -= np.dot(f1.T, db1)

        self.biases[0] -= np.dot(vec_ones, db0)
        self.biases[1] -= np.dot(vec_ones, db1)


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    X_train = scale(X_train)
    X_test = scale(X_test)

    model = TwoLayerNeural(np.shape(X_train)[1], 10)
    model.backprop(X_train[:2], y_train[:2], alpha=0.1)
    y_pred = model.forward(X_train[:2])
    print(mse(y_pred, y_train[:2]).flatten().tolist())
