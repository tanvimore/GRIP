import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#about the dataset
ds = pd.read_csv("http://bit.ly/w-data")
print("First 5 entries:\n{}" .format(ds.head()))
print("\n(rows,columns): {}".format(ds.shape))
print("\nDescription about the dataset:\n{}".format(ds.describe()))

#Visualization
plt.scatter(ds['Hours'], ds['Scores'])
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Scores')
plt.show()

#train-test-split
x = ds.iloc[:, :-1].values
y = ds.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#training
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#plotting regression line
line = regressor.coef_*x + regressor.intercept_
plt.scatter(x, y)
plt.plot(x, line, color='magenta')
plt.xlabel('Hours studied')
plt.ylabel('Scores')
plt.show()

#prediction
y_pred = regressor.predict(x_test)
print("\nPredicted values are:\n{}".format(y_pred))

#visualising the training set data
plt.scatter(x_train, y_train, color = 'cyan')
plt.plot(x_train, regressor.predict(x_train), color = 'yellow')
plt.title('Hours vs Percentage (Predicted)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scored')
plt.show()

#comparing actual values with predicted ones
comparison = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print("\nComparison between Actual & Predicted values:\n{}".format(comparison))

#predicting score
comparison = np.array(9.25)
comparison = comparison.reshape(-1, 1)
pred = regressor.predict(comparison)
print("\nIf the student studies for 9.25 hours/day, the score is {}.".format(pred))
