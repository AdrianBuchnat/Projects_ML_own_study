import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

cars = pd.read_csv('auto-mpg.csv')

print(cars.head())

X = cars.iloc[:, 1:8]
X = X.drop('horsepower', axis = 1)
y = cars.loc[:, 'mpg']

print(X.head())
print(y.head())

lr = LinearRegression()
lr.fit(X.to_numpy(), y)
print(lr.score(X.to_numpy(), y))

my_car1 = [4, 160, 190, 12, 90, 1]
my_car2 = [4, 200, 260, 15, 83, 1]
     
cars = [my_car1, my_car2]

mpg_predict = lr.predict(cars)
print(mpg_predict)