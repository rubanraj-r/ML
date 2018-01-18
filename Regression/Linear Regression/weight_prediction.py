#                                                                          #
# this model predicts the WEIGHT of a person from the HEIGHT of the person #
#                                                                          #

from sklearn.linear_model import LinearRegression

ht = [[152], [166], [174], [179], [183], [172]]
wt = [62, 64, 85, 72, 78, 68]

lm = LinearRegression()
lm.fit(ht, wt)
prediction = lm.predict(166)
coef = lm.coef_
intercept = lm.intercept_
print(prediction, ' ', intercept, ' ', coef)