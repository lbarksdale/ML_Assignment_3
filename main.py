# Created by Levi Barksdale in Oct 2023 for Math 6335 - Machine Learning and Control Theory
# Homework 3
import math

# This file trains a model to verify the Robbins-Monro theorem


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Defines the function specific to this problem
def f(x, p, g, a):
    return g * np.linalg.norm(x) ** 2 / 2 + x * p * math.pow((1 + np.linalg.norm(x) ** 2 / 2), a)


def Y(x, p, k):
    Y = np.zeros(len(x))
    for i in range(len(x)):
        unit_vector = np.zeros(len(x))
        unit_vector[i] = 1
        right_point = np.linalg.norm(x + unit_vector * al(k)) ** 2
        left_point = np.linalg.norm(x - unit_vector * al(k)) ** 2
        Y[i] = 5 * (right_point - left_point) + getX(x, p) * ((1 + right_point / 2) ** (1/2) - (1 + left_point / 2) ** (1/2))
    return Y / (2 * al(k))


# Function that returns alpha_k based on index k. Different from constant alpha given later.
def al(k):
    return 1/k


# Iterates one step
def iter(x, p, k):
    return x - al(k) * Y(x, p, k)


def getX(x, p):
    total = 0
    for i in range(len(x)):
        total = total + x[i] * p[i]
    return total


# Constants used in the function f
alpha = 1 / 2
gamma = 10

# Values that define distribution
x = [1, 2, 3]
p = [1 / 3, 1 / 3, 1 / 3]

# Creates discrete distribution. This is honestly just used for plotting.
dist = stats.rv_discrete(name='dist', values=(x, p))

fig, ax = plt.subplots(1, 1)
ax.plot(x, dist.pmf(x), 'ro', ms=12, mec='r')
ax.vlines(x, 0, dist.pmf(x), colors='r', lw=4)
plt.show()

num_iterations = 100

errors = np.zeros(num_iterations + 1)
errors[0] = np.log(np.linalg.norm(x))

for i in range(num_iterations):
    x = iter(x, p, i + 1)
    errors[i + 1] = np.log(np.linalg.norm(x))

plt.plot(errors)
plt.show()
