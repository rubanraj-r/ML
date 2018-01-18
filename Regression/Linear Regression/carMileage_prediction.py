#                                                  #
# Car mileage prediction using multiple parameters #
#                                                  #

import pandas as pd
from sklearn.linear_model import LinearRegression

dict = {
	'mileage' : pd.Series([23.0, 18.5, 19.3, 21.5, 17.0]),
	'cc' : pd.Series([1010, 1280, 1280, 795, 1461]),
	'hp' : pd.Series([88, 94, 95, 84, 108]),
	'wt' : pd.Series([1200, 1350, 1400, 850, 1550])
}

df = pd.DataFrame(dict)

lm = LinearRegression()
lm.fit(df[['cc', 'hp', 'wt']], df['mileage'])

print(lm.predict([[1000, 95, 1008]]))
print(lm.intercept_, lm.coef_, lm.score(df[['cc', 'hp', 'wt']], df['mileage']))