import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 1: Print the values for x and y
print(x)
print(y)
# Step 2: Standardize the data using StandardScaler, 
m = StandardScaler().fit(x)
# Step 3: Transform the data
x = m.transform(x)
# Step 4: Split the data into training and testing data
xtrn, xtst, ytrn, ytst = train_test_split(x, y)
# Step 6: Create a LogsiticRegression object and fit the data
model = linear_model.LogisticRegression().fit(xtrn, ytrn)
# Step 7: Print the score to see the accuracy of the model
score=model.score(xtst,ytst)
print(score)
# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data
for i in range(len(xtst)):
    x=m.inverse_transform(xtst)
    if x[i][2]==0:
        Gender="Male"
    else:
        Gender="Female"
    if ytst[i]!=y[i]:
        print(f"Gender:{x[i][2]} Salary:{int(x[i][1])} Age:{int(x[i][0])} Purchased(predicted):{y[i]} Purchased(real):{ytst[i]}")