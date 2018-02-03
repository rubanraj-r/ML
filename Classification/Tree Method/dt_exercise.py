#                                                     #
# Decision Tree and Random Forest Classifier Exercise #
#                                                     #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('data/kyphosis.csv')

# Visualizing the data
sns.pairplot(data, hue = 'Kyphosis')
plt.show()

# Train test split
X = data.drop('Kyphosis', axis = 1)
y = data['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 15)

# Decision Tree Classifier model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(predictions)

# prediction metrics for Decision Tree
confusion_matrix = confusion_matrix(y_test, predictions)
classification_report = classification_report(y_test, predictions)
print('DecisionTreeClassifier')
print(confusion_matrix)
print(classification_report)

# Random Forest Classifier Model
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(rfc_pred)

# prediction metrics for Random Forest Classifier
rfc_conf_mat = confusion_matrix(y_test, rfc_pred)
rfc_cls_report = classification_report(y_test, rfc_pred)
print('RandomForestClassifier')
print(rfc_conf_mat)
print(rfc_cls_report)