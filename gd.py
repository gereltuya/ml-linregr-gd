import numpy as np
import pandas as pd

def compute_cost(m, w0, w1, x, y):
    return 1/m * sum([(w0 + w1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=1500):
    converged = False
    iter = 0
    m = x.shape[0]

    w0 = 0
    w1 = 0
    J = compute_cost(m, w0, w1, x, y)
    num_iter = 0

    print('\n---Gradient descent started')
    print('\nFirst weights: weight0 = ' + str(w0)+'; weight1 = '+str(w1))
    print('\nFirst error value: J =', J[0]);

    while not converged:
        grad0 = 1.0/m * sum([(w0 + w1*np.asarray([x[i]]) - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(w0 + w1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])

        temp0 = w0 - alpha * grad0
        temp1 = w1 - alpha * grad1

        w0 = temp0
        w1 = temp1

        e = compute_cost(m, w0, w1, x, y)
        # print('J = ', e)

        if abs(J - e) < 0.0001:
            print('\n---Gradient descent stopped: Too little difference in error!')
            converged = True

        J = e
        iter += 1

        if iter == max_iter:
            print('\n---Gradient descent stopped: Max interactions exceeded!')
            converged = True

    print('\nIteration count: num_iter =', num_iter);
    print('\nLast error value: J =', J[0]);

    return w0, w1

if __name__ == '__main__':

    df = pd.read_csv('data/test_2d.txt', names=['x','y'])
    x = df['x']
    y = df['y']

    alpha = 0.01
    ep = 0.01

    weight0, weight1 = gradient_descent(alpha, x, y, ep, max_iter=1500)
    print('\nLast weights: weight0 = ' + str(weight0)+'; weight1 = '+str(weight1))
    print('\n')
