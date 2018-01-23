#                                                    #
# Titanic survivor prediction for Kaggle Competition #
#                                                    #

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

train_raw = pd.read_csv('Titanic Data/train.csv')
test_raw = pd.read_csv('Titanic Data/test.csv')

# dropping cabin column - too many nan
train = train_raw.drop('Cabin', axis = 1)
test = test_raw.drop('Cabin', axis = 1)


def impute_age(cols):
	Age = cols[0]
	Pclass = cols[1]
	
	if pd.isnull(Age) :
		if Pclass == 1 :
			return 38
		elif Pclass == 2 :
			return 29
		elif Pclass == 3 :
			return 24
	else :
		return Age

# fixing na using mean age value with respect to pclass
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis = 1)

# drop rest na values
train = train.dropna()
test = test.dropna()

# getting dummy value for sex column
sex_train = pd.get_dummies(train['Sex'], drop_first = True)
sex_test = pd.get_dummies(test['Sex'], drop_first = True)

# getting dummy value for embarked column
embark_train = pd.get_dummies(train['Embarked'], drop_first = True)
embark_test = pd.get_dummies(test['Embarked'], drop_first = True)

# getting dummy values for Pclass column
Pclass_train = pd.get_dummies(train['Pclass'], drop_first = True)
Pclass_test = pd.get_dummies(test['Pclass'], drop_first = True)

# concatenating the columns
train = pd.concat([train, sex_train, embark_train, Pclass_train], axis = 1)
test = pd.concat([test, sex_test, embark_test, Pclass_test], axis = 1)

# dropping the unwanted columns
train = train.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis = 1)
test = test.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis = 1)

# splitting the training and testing data
X_train = train.drop(['Survived'], axis = 1)
y_train = train['Survived']

X_test = test

# logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
predictions = log_model.predict(X_test)

result_dict = {
	'PassengerId' : pd.Series(test_raw['PassengerId']),
	'Survived' : pd.Series(predictions)
}

result = pd.DataFrame(result_dict)

def sur(cols):
	if pd.isnull(cols[0]) :
		return 0
	else :
		return cols[0]

result['Survived'] = result[['Survived']].apply(sur, axis = 1)
try:
	result.to_csv('Titanic Data/result.csv', index=False)
	print("Successfully written to 'Titanic Data/result.csv'")
except Exception as e:
	print("can't write to a csv file - > ", e)