#                                                              #
# using the iris data set for understanding classifier problem #
#                                                              #

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = DecisionTreeClassifier()
clf.fit(train_data, train_target)
prediction = clf.predict(test_data)
print('predicted value - > \n', prediction, '\n Score - > \n', clf.score(train_data, train_target))
