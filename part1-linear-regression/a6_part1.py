import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values


x = x.reshape(-1,1)

model= LinearRegression().fit(x,y)
coef= (round(float(model.coef_[0])),2)
intercept = round(float(model.intercept), 2)
r_squared = model.score(x,y)
x_predict = 43
prediction = model.predict([[x_predict]])


print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")
print(f"Prediction when x is {x_predict}: {prediction}")


plt.figure(figsize=(6,4))

plt.scatter(x,y, c="purple")
plt.scatter(x_predict, prediction, c="blue")

plt.xlabel("Temperature °F")
plt.ylabel("Chirps per Minute")
plt.title("Cricket Chirps by Temperature")

plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend()
plt.show()