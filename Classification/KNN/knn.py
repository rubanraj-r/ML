#                             #
# K Nearest Neighbor Exercise #
#                             #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('data/Classified Data', index_col = 0)

# normalizing features using scikit scaler
scaler = StandardScaler()
scaler.fit(data.drop('TARGET CLASS', axis = 1))
scaled_features = scaler.transform(data.drop('TARGET CLASS', axis = 1))

# create dataframe using scaled values
scaled_data = pd.DataFrame(scaled_features, columns = data.columns[:-1])

# data for ML
X = scaled_data
y = data['TARGET CLASS']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# KNN Algorithm
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Analyzing the prediction report
confusion_matrix = confusion_matrix(y_test, predictions)
classification_report = classification_report(y_test, predictions)

print(confusion_matrix)
print(classification_report)

# Analyzing the error rate by differing k(n_neighbors) value
error_rate = []
for i in range(1, 40):
	knn = KNeighborsClassifier(n_neighbors = i)
	knn.fit(X_train, y_train)
	pred_i = knn.predict(X_test)
	error_rate.append(np.mean(pred_i != y_test))

# visualizing the error_rate with respect to n_neighbors value
plt.plot(range(1, 40), error_rate, marker = 'o')
# plt.show()

# The visualization inference reveals that k = 17 is better 