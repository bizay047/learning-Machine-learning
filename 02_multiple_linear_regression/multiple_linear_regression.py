import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import darwinVersionString
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("StudentPerformanceFactors.csv")
# print(f"Rows, Columns: {df.shape}\n")

# df.info()
# df.describe()

features = ["Hours_Studied", "Attendance", "Sleep_Hours" ,"Previous_Scores" ,"Physical_Activity"]
target = "Exam_Score"
X=df[features]
y = df[target]

X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split Data into training and testing sets
X_train, X_test, y_train ,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with the mulitple linear regression Model
model =LinearRegression()
model.fit(X_train, y_train)

#model Coefficeints and intercep
print("Intercept:\n ", model.intercept_)
coeff_df=pd.DataFrame({
    "Coefficient" :model.coef_,
    "Feature" :X.columns
})
coeff_df

#make predication
y_pred = model.predict(X_test)

# Evaluate Model PErformance
mse = mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print("Mean Squared Error(MSE): \n", mse)
print("R2 Score:\n ", r2)

plt.figure(figsize=(8, 6))

# Scatter plot for predicted values
plt.scatter(
    y_test,
    y_pred,
    color='green',


    label='Predicted Values',
    alpha=0.5
)

# Ideal line (Actual = Predicted)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='blue',
    linewidth=2,
    label='Ideal Fit (Actual = Predicted)'
)
# Best-fit line using linear regression on y_test vs y_pred
m, b = np.polyfit(y_test, y_pred, 1)

plt.plot(
    y_test,
    m * y_test + b,
    linewidth=2,
    color='red',
    linestyle='dashed',
    label="Best Fit Line"
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Student Performance")
plt.legend()
plt.grid(True)

plt.show()


#observation
print(
    "The scatter plot shows a positive correlation between actual and predicted values, \n"
    "indicating that the Multiple Linear Regression model performs reasonably well.\n"
)
#
print(
    "Multiple Linear Regression was successfully implemented to predict student performance\n "
    "using study hours, attendance, and assignment scores.\n "
    "The model performance was evaluated using Mean Squared Error and R² score,\n "
    "demonstrating effective prediction capability.\n"
)
