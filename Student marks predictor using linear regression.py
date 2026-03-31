import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Sleep": [6, 6.5, 7, 6, 7.5, 8, 7, 8, 7.5, 8],
    "Attendance": [60, 65, 70, 72, 75, 80, 85, 88, 90, 95],
    "Marks": [35, 40, 50, 55, 65, 70, 75, 85, 90, 95]
}

df = pd.DataFrame(data)

X = df[["Hours", "Sleep", "Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

hours = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
attendance = float(input("Enter attendance (%): "))

predicted_marks = model.predict([[hours, sleep, attendance]])

print(f"Predicted Marks: {predicted_marks[0]:.2f}")

plt.scatter(df["Hours"], y)
plt.plot(df["Hours"], model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()