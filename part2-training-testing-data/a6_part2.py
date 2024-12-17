import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values
print(x)
x = x.reshape(-1,1)
print(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)
model = LinearRegression().fit(xtrain, ytrain)
coef = round(float(model.coef_[0]), 2)
intcpt = round(float(model.intercept_), 2)
rSquared = model.score(xtrain,ytrain)
equation=coef*x + intcpt
print(f"Equation: y = {coef}x + {intcpt}, r^2:{rSquared}")
predict = model.predict(xtest)
predict = np.around(predict, 2)
print("\nTesting Linear Model with Testing Data:")
for i in range(len(xtest)):
    actual=ytest[i]
    predicted=predict[i]
    x=xtest[i]
    dist=actual-predicted
    print(f"x value:{float(x[0])} Predicted y value: {predicted} Actual y value: {actual} distance: {abs(round(dist,2))}")
print("\nTesting Linear Model with Testing Data:")
plt.figure(figsize=(5,4))
plt.scatter(xtrain,ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")
plt.scatter(xtest, predict, c="red", label="Predictions")
plt.xlabel("Age")
plt.ylabel("Blood pressure")
plt.title("Blood pressure VS Age")
plt.plot(xtrain, coef*xtrain + intcpt, c="b", label="Line of Best Fit")
plt.legend()
plt.show()