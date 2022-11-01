import numpy as np
import pandas as pd


def get_error(x, y, w):
    return np.sum(((x @ w - y) ** 2), axis=0) / x.shape[0]


def get_gradient(x, y, w):
    return (2 / x.shape[0]) * (x.T @ (x @ w - y))


def gradient_descent(x, y, alpha=0.001, ep=0.0001, max_iter=1500):
    n = x.shape[1]
    m = x.shape[0]

    w = np.zeros((n + 1, 1))
    x = np.append(np.ones((m, 1)), x, axis=1)  # shaped (97, 2)

    converged = False
    i = 0
    J = get_error(x, y, w)

    print("\n---Gradient descent started")
    print("\nFirst weights: w = " + str(w))
    print("\nFirst error value: J =", J)

    while not converged:
        w = w - alpha * get_gradient(x, y, w)
        e = get_error(x, y, w)

        if abs(J - e) < 0.01:
            print("\n---Gradient descent stopped: Too little difference in error!")
            converged = True

        i += 1
        J = e

        if i == max_iter:
            print("\n---Gradient descent stopped: Max interactions exceeded!")
            converged = True

    print("\nIteration count: i =", i)
    print("\nLast error value: J =", J)

    return w


df = pd.read_csv("data/test_2d.txt", names=["x", "y"])
x = np.array([df["x"]]).T
y = np.array([df["y"]]).T

w = gradient_descent(x, y)

print("\nLast weights: w = " + str(w))
print("\n")
