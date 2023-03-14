from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
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

par_knn = {"n_neighbors" : [3, 4], "weights" : ['uniform', 'distance'], 'algorithm' : ['auto', 'brute']}
par_rfc = {"n_estimators" : [300, 500], "max_features" : ['auto', 'log2'], "class_weight" : ['balanced', 'balanced_subsample'], "random_state" : [40]}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid=par_knn, scoring='accuracy', n_jobs=-1)
grid_rfc = GridSearchCV(RandomForestClassifier(), param_grid=par_rfc, scoring='accuracy', n_jobs=-1)

grid_knn.fit(X_train_norm, y_train)
grid_rfc.fit(X_train_norm, y_train)

print(f'K-nearest neighbours algorithm\nbest estimator: {grid_knn.best_estimator_}')
print(f'accuracy: {accuracy_score(y_test, grid_knn.best_estimator_.predict(X_test_norm))}')

print(f'Random forest algorithm\nbest estimator: {grid_rfc.best_estimator_}')
print(f'accuracy: {accuracy_score(y_test, grid_rfc.best_estimator_.predict(X_test_norm))}')
