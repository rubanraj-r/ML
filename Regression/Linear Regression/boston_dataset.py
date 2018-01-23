#                                        #
# Linear Regression using Boston Dataset #
#                                        #

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import metrics

boston = load_boston()
X = boston.data
y = boston.target

# splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print(pd.DataFrame(predictions, y_test, columns = ['test data']))
print('\n Prediction Score - > ', lm.score(X_train, y_train))

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print('\n MAE : ', mae, '\n MSE : ', mse, '\n RMSE : ', rmse)


