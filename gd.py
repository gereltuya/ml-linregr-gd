import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets.samples_generator import make_regression

def compute_cost_function(m, t0, t1, x, y):
  return 1/2/m * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=1500):
    converged = False
    iter = 0
    m = x.shape[0]

    t0 = 0
    t1 = 0

    J = compute_cost_function(m, t0, t1, x, y)
    print('J=', J);
    num_iter = 0
    while not converged:
        grad0 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])

        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        t0 = temp0
        t1 = temp1

        e = compute_cost_function(m, t0, t1, x, y)
        print ('J = ', e)
        J = e
        iter += 1

        if iter == max_iter:
            print ('Max interactions exceeded!')
            converged = True

    return t0,t1

def plot_cost_function(x, y, m):
    t0 = list(range(0, x.shape[0]))
    j_values = []
    for i in range(len(t0)):
        j_values.append(compute_cost_function(m, i, i, x, y)[0])
    print ('j_values', len(j_values), len(x), len(y))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, j_values, label='parametric curve')
    ax.legend()
    plt.show()

if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                        random_state=0, noise=35)

    alpha = 0.01
    ep = 0.01

    theta0, theta1 = gradient_descent(alpha, x, y, ep, max_iter=1500)
    print ('theta0 = ' + str(theta0)+' theta1 = '+str(theta1))
