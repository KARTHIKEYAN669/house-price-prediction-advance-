import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

data={
    "Size":[800,1200,1500,1800,2000,2200,2500,2700,3000,3200],
    "Bedrooms":[2,3,3,4,4,5,5,6,6,7],
    "Age":[20,15,10,8,5,3,2,1,1,1],
    "Distance_to_City":[15,10,8,6,5,4,3,3,2,2],
    "Price":[150000,200000,250000,300000,350000,400000,450000,500000,550000,600000]
}

df=pd.DataFrame(data)
print(df)

x=df[["Size","Bedrooms","Age","Distance_to_City"]]
y=df["Price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

model1=DecisionTreeRegressor()
model1.fit(x_train,y_train)

y_pred_lr=model.predict(x_test)
print("LR Predictions:",y_pred_lr)

y_pred_dt=model1.predict(x_test)
print("DT Predictions:",y_pred_dt)

print("Linear Regression MAE:",mean_absolute_error(y_test,y_pred_lr))
print("Linear Regression MSE:",mean_squared_error(y_test,y_pred_lr))

print("Decision Tree MAE:",mean_absolute_error(y_test,y_pred_dt))
print("Decision Tree MSE:",mean_squared_error(y_test,y_pred_dt))

new_data=pd.DataFrame([[1800,4,7,6]],columns=["Size","Bedrooms","Age","Distance_to_City"])
print("Prediction Price(LR):",model.predict(new_data))
print("Prediction Price(DT):",model1.predict(new_data))
print("Feature Importance:",model1.feature_importances_)
print("\n---Model Comparison---")


print("Actual:",list(y_test))
print("Predicted (LR):",y_pred_lr)
print("Predicted (DT):",y_pred_dt)

plt.scatter(y_test,y_pred_lr)
plt.plot([y.min(),y.max()],[y.min(),y.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression")
plt.show()

plt.scatter(y_test,y_pred_dt)
plt.plot([y.min(),y.max()],[y.min(),y.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Decision Tree")
plt.show()