import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

model= LinearRegression().fit(x,y)
coef= (round(float(model.coef_[0])),2)
intercept = round(float(model.intercept), 2)
r_squared = model.score(x,y)