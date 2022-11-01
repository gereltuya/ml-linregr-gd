# Repository structure

* [README](https://github.com/gereltuya/ml-linregr-gd/blob/main/README.md) contains Task completion checklist.
* [Jupyter notebook](https://github.com/gereltuya/ml-linregr-gd/blob/main/ml-linregr-gd.ipynb) contains the final state of all the code and results.
* [Python file](https://github.com/gereltuya/ml-linregr-gd/blob/main/ml-linregr-gd.py) contains exactly the same code content as the notebook above.
* [Requirement list](https://github.com/gereltuya/ml-linregr-gd/blob/main/requirements.txt) is self-explanatory.
* [Data folder](https://github.com/gereltuya/ml-linregr-gd/tree/main/data) contains 2 different datasets used.
* [Images folder](https://github.com/gereltuya/ml-linregr-gd/tree/main/images) contains resulting images from analysis, training and testing.
* [Modules folder](https://github.com/gereltuya/ml-linregr-gd/tree/main/modules) contains 2 raw files used while completing this homework.

# Task 1. Linear Regression, Gradient Descent Method

Task explanation taken from [here](https://github.com/rustam-azimov/ml-course/blob/main/tasks/task01_linregr_gd.md).

* **Deadline**: 31.10.2022, 23:59
* Main full score: 5
* Maximum score: 11

## Task

- [x] Independently implement the function **gradient_descent(X, y)**, which trains a linear regression model for a given training sample, optimizing the functional using the gradient descent method (Batch Gradient Descent, GD) and returning the weight vector **w**. As a functional, one can choose, for example, the error function **MSE** + $L_2$-regulator. Use matrix-vector operations to calculate the gradient.
- [x] Find data on which it will be interesting to solve the regression problem. The dependence of the target feature on the non-target ones should not be too complicated so that the trained linear model can show an acceptable result. As a last resort, take data to predict the cost of cars [here](https://github.com/rustam-azimov/ml-course/tree/main/data/car_price) (target feature for prediction --- **price** ).
- [x] Read data, perform initial data analysis, perform Data Cleaning if necessary.
- [x] Perform exploratory analysis (EDA), use visualization, draw conclusions that may be useful in further solving the regression problem.
- [x] If necessary, perform useful data transformations (for example, transform categorical features into quantitative ones), remove unnecessary features, create new ones (Feature Engineering).
- [x] Randomly split the data into training and test sets using the methods of existing libraries.
- [x] Use data scaling when training models.
- [x] Train the model on the training set using the **gradient_descent(X, y)** function. Assess model quality on training and test sets using **MSE**, **RMSE** and $R^2$.
- [x] Train the model using the existing library. For example, in **sklearn** you can use **Ridge** for the $L_2$ regularizer. Compare the quality with your implementation.
- [ ] Repeat the same, but using cross-validation.
- [ ] Create a table, with rows (mse-train, mse-test, rmse-train, rmse-test, r2-train, r2-test) and columns (Fold1, Fold2, ..., Foldk, E, STD) , where k --- number of folds in cross-validation, E --- mat. expectation and STD --- standard deviation. To conclude.
- [x] * (+1 point) Before training models, select the best number (and the subset itself) of features, for example, using Recursive Feature Elimination (RFE).
- [x] * (+1 point) In all your implementations, add the ability to adjust the necessary hyperparameters, and in the process of training **all** models, select the optimal values ​​of these hyperparameters.
- [ ] * (+2 points) Also independently implement the Stochastic Gradient Descent (SGD) method, train the models and add them to all comparisons.
- [ ] * (+2 points) Also independently implement the Mini Batch Gradient Descent method, train the models and add them to all comparisons.
