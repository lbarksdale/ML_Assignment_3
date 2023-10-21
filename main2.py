# Created by Levi Barksdale for Math 6335 - Machine Learning and Control Theory
# Assignment 3
# Problem 2

import matplotlib.pyplot as plt
import numpy as np


# Defines sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid function
def sigder(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Function used in neural network
def g(x):
    return x


# Derivative of g
def gder(x):
    return 1


def F(A, b, x, y):
    return pow(np.abs(g(sigmoid(A * x + b))), 2) + gamma * (pow(A, 2) + pow(b, 2))


def f(x, A, b):
    return sigmoid(A * x + b)


# Function to get error between guess and actual solution
def error(A, b, x):
    return ((A * x + b) - y_vals(x))**2


def deriv_error(A, b, x):
    return 2 * ((A * x + b) - y_vals(x))


# Model training parameters
num_training_points = 300
training_interval_width = 10

# Declare function constants
alpha = 1
beta = 2
gamma = 3
d = 4
lda = 1
mu = 2
nu = 2


# Gets y values for specified function. This function may be changed to change the function that is being approximated
def y_vals(sample_xvals):
    return (alpha * pow(sample_xvals, 3) + beta * pow(sample_xvals, 2) + gamma * sample_xvals + d) / (
            lda * pow(sample_xvals, 2) + mu * sample_xvals + nu)


# Create training data
sample_xvals = np.linspace(0, training_interval_width, num_training_points)
sample_yvals = y_vals(sample_xvals)

# initial guesses
A = 1
b = 1

# Plot true function
plt.plot(sample_xvals, sample_yvals)
plt.show()

cur_error = error(A, b, sample_xvals)
learn_rate = 1
num_iterations = 1000


"""The issue is that the derivative of the sigmoid function is always positive, so it means that A is only decreasing
Same for b
"""

for j in range(num_iterations):
    for i in range(num_training_points):
        # gradient = sigder(f(A, b, sample_xvals[i]))
        # gradient = sigder(A * sample_xvals[i] + b)
        # gradient = sigder(deriv_error(A, b, sample_xvals[i]))
        # gradient = sigder(A * sample_xvals[i] + b - y_vals(sample_xvals[i]))
        gradient = deriv_error(A, b, sample_xvals[i]) / num_training_points
        A = A - learn_rate * gradient * sample_xvals[i]
        b = b - learn_rate * gradient
        cur_error = error(A, b, sample_xvals)
        learn_rate = 1 / (j + 1)

print(A)
print(b)

# plt.plot(sample_xvals, f(A, b, x))
plt.plot(sample_xvals, A * sample_xvals + b)
plt.show()
