from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import Normalizer
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
x = np.concatenate((X_train, X_test))[:6000]
y = np.concatenate((y_train, y_test))[:6000]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

normalizer = Normalizer()

X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

models = {
    'K-nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=40),
    'Logistic Regression': LogisticRegression(solver="liblinear", random_state=40),
    'Random Forest': RandomForestClassifier(random_state=40)
}


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)
    score = accuracy_score(target_test, y_pred)
    print(f'Model: {model.__class__.__name__}\nAccuracy: {score:.3f}\n')
    return score


scores = {model.__class__.__name__ : fit_predict_eval(model, X_train_norm, X_test_norm, y_train, y_test) for model in models.values()}

print("The answer to the 1st question: yes")

print("The answer to the 2nd question:", ", ".join([model[0] + "-" + str(round(model[1], 3)) for model in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:2]))
