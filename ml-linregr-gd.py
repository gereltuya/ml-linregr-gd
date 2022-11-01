#!/usr/bin/env python
# coding: utf-8

# - [x] Самостоятельно реализовать функцию **gradient_descent(X, y)**, которая по заданной обучающей выборке обучает модель линейной регрессии, оптимизируя функционал методом градиентного спуска (Batch Gradient Descent, GD) и возвращая вектор весов **w**. В качестве функционала можно выбрать, например, функцию ошибок **MSE** + $L_2$-регуляризатор. Использовать матрично-векторные операции для вычисления градиента.

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def get_error(x, y, w):
    return np.sum(((x @ w - y) ** 2), axis=0) / x.shape[0]


# In[3]:


def get_gradient(x, y, w):
    return (2 / x.shape[0]) * (x.T @ (x @ w - y))


# In[4]:


def gradient_descent(x, y, alpha=0.001, ep=0.0001, max_iter=1500, err_tol=0.01):
    n = x.shape[1]
    m = x.shape[0]

    w = np.zeros((n + 1, 1))
    x = np.append(np.ones((m, 1)), x, axis=1)

    converged = False
    i = 0
    J = get_error(x, y, w)

    print("\n---Gradient descent started")
    print("\nFirst weights: w = " + str(w))
    print("\nFirst error value: J =", J)

    while not converged:
        w = w - alpha * get_gradient(x, y, w)
        e = get_error(x, y, w)

        if abs(J - e) < err_tol:
            print("\n---Gradient descent stopped: Too little difference in error!")
            converged = True

        i += 1
        J = e

        if i == max_iter:
            print("\n---Gradient descent stopped: Max interactions exceeded!")
            converged = True

    print("\nIteration count: i =", i)
    print("\nLast error value: J =", J)
    print("\nLast weights: w = " + str(w))
    print("\n")

    return w


# In[5]:


df_0 = pd.read_csv("data/test_2d.txt", names=["x", "y"])
x = np.array([df_0["x"]]).T
y = np.array([df_0["y"]]).T

w = gradient_descent(x, y)


# - [x] Find data on which it will be interesting to solve the regression problem. The dependence of the target feature on the non-target ones should not be too complicated so that the trained linear model can show an acceptable result. As a last resort, take data to predict the cost of cars [here](https://github.com/rustam-azimov/ml-course/tree/main/data/car_price) (target feature for prediction --- **price** ).
# - [x] Read data, perform initial data analysis, perform Data Cleaning if necessary.
# - [x] * (+1 point) Before training models, select the best number (and the subset itself) of features, for example, using Recursive Feature Elimination (RFE).

# In[6]:


from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[7]:


df = pd.read_csv('data/car_sales.csv')


# In[8]:


df


# In[9]:


sns.pairplot(df)


# In[10]:


sns.heatmap(df.corr())


# - [x] Perform exploratory analysis (EDA), use visualization, draw conclusions that may be useful in further solving the regression problem.
# - [x] If necessary, perform useful data transformations (for example, transform categorical features into quantitative ones), remove unnecessary features, create new ones (Feature Engineering).

# In[11]:


# 1. Which brands are sold the best?

total_sales = df['Price_in_thousands']*df['Sales_in_thousands']
df = df.assign(Total_sales_in_millions=total_sales)

df_1 = df.groupby(['Manufacturer']).sum()[['Sales_in_thousands', 'Total_sales_in_millions']]
df_1.columns = ['Sales_in_thousands', 'Total_sales_in_millions']

plt.figure(figsize=(16,9))
plt.scatter(df_1['Total_sales_in_millions'], df_1['Sales_in_thousands'], df_1['Total_sales_in_millions']/10)

for i, label in enumerate(df_1.index):
    plt.annotate(label, (df_1['Total_sales_in_millions'][i], df_1['Sales_in_thousands'][i]))

plt.title('1. Which brands are sold the best?')
plt.xlabel('Total sales in millions')
plt.ylabel('Sales in thousands')


# In[12]:


# Which vehicle type/engine size pairs are the most expensive on average?

df_2 = df.groupby(['Vehicle_type', 'Engine_size']).mean()[['Price_in_thousands']]
df_2.columns = ['Price_in_thousands_avg']
df_2.reset_index(inplace=True)

plt.figure(figsize=(16,9))
plt.scatter(df_2['Engine_size'], df_2['Price_in_thousands_avg'], df_2['Price_in_thousands_avg']*20)

for i, label in enumerate(df_2['Vehicle_type']):
    plt.annotate(label, (df_2['Engine_size'][i], df_2['Price_in_thousands_avg'][i]))

plt.title('Which vehicle type/engine size pairs are the most expensive on average?')
plt.xlabel('Engine size')
plt.ylabel('Price in thousands avg')


# In[13]:


# 3. Which cars have the least density with respect to width/length-based volume and curb weight?

car_names = df['Manufacturer'].str.cat(df['Model'], sep=', ')
df = df.assign(Car_name=car_names)

car_density = df['Curb_weight']/(df['Width']*df['Length'])
df = df.assign(Car_density=car_density)

df_3 = df.sort_values(by='Car_density', ascending=True)

plt.figure(figsize=(16,9))
plt.bar(df_3['Car_name'][:10], df_3['Car_density'][:10])
plt.title('3. Which cars have the least density with respect to width/length-based volume and curb weight?')


# In[14]:


# 4. Which cars are the best in terms of fuel efficiency and the price?

fuel_efficiency_price = df['Fuel_efficiency']/df['Price_in_thousands']
df = df.assign(Fuel_efficiency_price=fuel_efficiency_price)

df_4 = df.sort_values(by='Fuel_efficiency_price', ascending=False)

plt.figure(figsize=(16,9))
plt.bar(df_4['Car_name'][:10], df_4['Fuel_efficiency_price'][:10])
plt.title('4. Which cars are the best in terms of fuel efficiency and the price?')


# - [x] Use data scaling when training models.
# 

# In[15]:


s = df_2["Price_in_thousands_avg"].to_numpy()
max_s = np.max(s)
min_s = np.min(s)
scaled = (s - min_s) / (max_s - min_s)

df_2_scaled = df_2
df_2_scaled["Price_in_thousands_avg_scaled"] = scaled
df_2_scaled


# - [x] Randomly split the data into training and test sets using the methods of existing libraries.

# In[16]:


df_2_train = df_2_scaled.sample(frac=0.8, random_state=25)
df_2_test = df_2.drop(df_2_train.index)


# - [x] Train the model on the training set using the **gradient_descent(X, y)** function. Assess model quality on training and test sets using **MSE**, **RMSE** and $R^2$.
# 
# - [x] * (+1 point) In all your implementations, add the ability to adjust the necessary hyperparameters, and in the process of training **all** models, select the optimal values ​​of these hyperparameters.

# In[17]:


x_train = np.array([df_2_train["Engine_size"]]).T
y_train = np.array([df_2_train["Price_in_thousands_avg_scaled"]]).T

w_train = gradient_descent(x_train, y_train, err_tol=0.0001)
y_pred_train = np.append(np.ones((x_train.shape[0], 1)), x_train, axis=1) @ w_train


# In[18]:


import matplotlib.pyplot as plt

plt.scatter(x_train, y_train,  color='black')
plt.plot(x_train, y_pred_train, color='blue', linewidth=3)
plt.show()


# In[19]:


x_test = np.array([df_2_test["Engine_size"]]).T
n_test = x_test.shape[0]
x_test = np.append(np.ones((n_test, 1)), x_test, axis=1)

y_test = np.array([df_2_test["Price_in_thousands_avg_scaled"]]).T
y_pred_test = x_test @ w_train

error_mse = np.sum(((y_pred_test - y_test) ** 2), axis=0) / n_test
error_rmse = np.sqrt(error_mse)
error_r2 = 1 - (error_mse * n_test) / np.sum(((y_pred_test - y_test.mean()) ** 2), axis=0)

print(error_mse, error_rmse, error_r2)


# - [x] Train the model using the existing library. For example, in **sklearn** you can use **Ridge** for the $L_2$ regularizer. Compare the quality with your implementation.
# 

# In[20]:


from sklearn.linear_model import Ridge

regr = Ridge(alpha=0.001, max_iter=1500, tol=0.01)
regr.fit(x_train, y_train)
y_pred_lib = regr.predict(x_train)

plt.scatter(x_train, y_train,  color='black')
plt.plot(x_train, y_pred_train, color='blue', linewidth=3)
plt.plot(x_train, y_pred_lib, color='red', linewidth=3)
plt.show()


# In[21]:


x_test_lib = np.array([df_2_test["Engine_size"]]).T
y_pred_test_lib = regr.predict(x_test_lib)

error_mse_lib = np.sum(((y_pred_test_lib - y_test) ** 2), axis=0) / n_test
error_rmse_lib = np.sqrt(error_mse_lib)
error_r2_lib = 1 - (error_mse_lib * n_test) / np.sum(((y_pred_test_lib - y_test.mean()) ** 2), axis=0)

print(error_mse_lib, error_rmse_lib, error_r2_lib)

