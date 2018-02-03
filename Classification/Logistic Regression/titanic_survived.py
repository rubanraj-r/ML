#                                                             #
# Classification using Logistic Regression using Titanic data #
#                                                             #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./Titanic Data/train.csv')
test = pd.read_csv('./Titanic Data/test.csv')

print(train.info())
sns.set_style('whitegrid')

# Heat map to figure out the missing data
sns.heatmap(train.isnull())
plt.show()

# count the number of passengers survived and dead on basis of passenger class
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
plt.show()

# Plot the age group of passengers on board
sns.distplot(train['Age'].dropna(), kde = False)     # kde false removes the curve in the plot
plt.show()

# Figuring out the siblings and spouse of the passengers
sns.countplot(x = 'SibSp', data = train)
plt.show()

# Prices paid to on board by the passengers
sns.distplot(train['Fare'], kde = False)
plt.show()

# Visualizing the average age group based on the passenger class
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
plt.show()

# Finding the mean age value of the passengers based on class
def impute_age(cols):
	Age = cols[0]
	Pclass = cols[1]
	
	if pd.isnull(Age) :
		
		if Pclass == 1 :
			return 37
		elif Pclass == 2 :
			return 29
		elif Pclass == 3 :
			return 24
			
	else :
		return Age
		
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)

train = train.drop(['Cabin'], axis = 1)

train = train.dropna()

print('train data - > ', train.head(5))
# sns.heatmap(train.isnull())
# plt.show()

sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)

train = pd.concat([train, sex, embark], axis = 1)

train = train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'], axis = 1)
print(train.head(5))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = train.drop('Survived', axis = 1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
predictions = log_model.predict(X_test)

print(predictions)
print(len(predictions))
print(metrics.accuracy_score(y_test, predictions))
print(log_model.score(X_test, y_test))
print(metrics.confusion_matrix(y_test, predictions))
