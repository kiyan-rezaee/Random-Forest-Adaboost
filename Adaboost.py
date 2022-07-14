import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics

data = datasets.load_digits()

X = data.images.reshape((len(data.images), -1))
Y = data.target

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

ada_clf = AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='rbf'), n_estimators=100)
ada_clf.fit(X[:1000], Y[:1000])

p = ada_clf.predict(X[1000:])
e = Y[1000:]
print(metrics.classification_report(e, p))

