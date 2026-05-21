# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Libraries
2. Import the data file and import numpy, matplotlib and scipy.
3. Visulaize the data and define the sigmoid function, cost function and gradient descent.
4. Plot the decision boundary .
5. Calculate the y-prediction

Program to implement the the Logistic Regression Using Gradient Descent.

**Developed by : SRIDHAR C <br>
RegisterNumber: 212225040425**
## Program:
```py


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Placement_Data.csv")

data1 = data.copy()

# Drop unnecessary columns
data1 = data1.drop(['sl_no', 'salary'], axis=1)

# Encoding categorical data
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Features and target
x = data1.iloc[:, :-1].values
y = data1["status"].values

# Feature scaling
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Initialize weights
theta = np.random.randn(x.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, x, y):
    h = sigmoid(x.dot(theta))
    epsilon = 1e-10
    return -np.mean(y*np.log(h+epsilon) + (1-y)*np.log(1-h+epsilon))

# Gradient Descent
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h - y) / m
        theta = theta - alpha * gradient
    return theta

# Training
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

# Prediction
def predict(theta, x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# Predictions
y_pred = predict(theta, x)

# Accuracy
accuracy = np.mean(y_pred == y)

print("Accuracy:", accuracy)
print("Predicted:\n", y_pred)
print("Actual:\n", y)

# New sample prediction
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])

# Normalize new data using training mean/std
xnew = (xnew - x.mean(axis=0)) / x.std(axis=0)

y_prednew = predict(theta, xnew)

print("Predicted Result:", y_prednew)

```

## Output:

<img width="826" height="360" alt="Screenshot 2026-05-14 113300" src="https://github.com/user-attachments/assets/485f46a0-f417-4709-b3ca-bd2338868052" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming is executed successfully.

