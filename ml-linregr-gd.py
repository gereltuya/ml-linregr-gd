#!/usr/bin/env python
# coding: utf-8

# - [x] Independently implement the function **gradient_descent(X, y)**, which trains a linear regression model for a given training sample, optimizing the functional using the gradient descent method (Batch Gradient Descent, GD) and returning the weight vector **w**. As a functional, one can choose, for example, the error function **MSE** + $L_2$-regulator. Use matrix-vector operations to calculate the gradient.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


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
# - [x] (+1 point) Before training models, select the best number (and the subset itself) of features, for example, using Recursive Feature Elimination (RFE).

# In[6]:


df = pd.read_csv('data/car_sales.csv')


# In[7]:


df


# In[8]:


sns.pairplot(df)


# In[9]:


sns.heatmap(df.corr())


# - [x] Perform exploratory analysis (EDA), use visualization, draw conclusions that may be useful in further solving the regression problem.
# - [x] If necessary, perform useful data transformations (for example, transform categorical features into quantitative ones), remove unnecessary features, create new ones (Feature Engineering).

# In[10]:


# 1. Which brands are sold the best?

total_sales = df['Price_in_thousands']*df['Sales_in_thousands']
df = df.assign(Total_sales_in_millions=total_sales)

df_1 = df.groupby(['Manufacturer']).sum()[['Sales_in_thousands', 'Total_sales_in_millions']]
df_1.columns = ['Sales_in_thousands', 'Total_sales_in_millions']

plt.figure(figsize=(24,13.5))
plt.scatter(df_1['Total_sales_in_millions'], df_1['Sales_in_thousands'], df_1['Total_sales_in_millions']/10)

for i, label in enumerate(df_1.index):
    plt.annotate(label, (df_1['Total_sales_in_millions'][i], df_1['Sales_in_thousands'][i]))

plt.title('1. Which brands are sold the best?')
plt.xlabel('Total sales in millions')
plt.ylabel('Sales in thousands')


# In[11]:


# Which vehicle type/engine size pairs are the most expensive on average?

df_2 = df.groupby(['Vehicle_type', 'Engine_size']).mean()[['Price_in_thousands']]
df_2.columns = ['Price_in_thousands_avg']
df_2.reset_index(inplace=True)

plt.figure(figsize=(24,13.5))
plt.scatter(df_2['Engine_size'], df_2['Price_in_thousands_avg'], df_2['Price_in_thousands_avg']*20)

for i, label in enumerate(df_2['Vehicle_type']):
    plt.annotate(label, (df_2['Engine_size'][i], df_2['Price_in_thousands_avg'][i]))

plt.title('Which vehicle type/engine size pairs are the most expensive on average?')
plt.xlabel('Engine size')
plt.ylabel('Price in thousands avg')


# In[12]:


# 3. Which cars have the least density with respect to width/length-based volume and curb weight?

car_names = df['Manufacturer'].str.cat(df['Model'], sep=', ')
df = df.assign(Car_name=car_names)

car_density = df['Curb_weight']/(df['Width']*df['Length'])
df = df.assign(Car_density=car_density)

df_3 = df.sort_values(by='Car_density', ascending=True)

plt.figure(figsize=(24,13.5))
plt.bar(df_3['Car_name'][:10], df_3['Car_density'][:10])
plt.title('3. Which cars have the least density with respect to width/length-based volume and curb weight?')


# In[13]:


# 4. Which cars are the best in terms of fuel efficiency and the price?

fuel_efficiency_price = df['Fuel_efficiency']/df['Price_in_thousands']
df = df.assign(Fuel_efficiency_price=fuel_efficiency_price)

df_4 = df.sort_values(by='Fuel_efficiency_price', ascending=False)

plt.figure(figsize=(24,13.5))
plt.bar(df_4['Car_name'][:10], df_4['Fuel_efficiency_price'][:10])
plt.title('4. Which cars are the best in terms of fuel efficiency and the price?')


# - [x] Use data scaling when training models.
# 

# In[14]:


s = df_2["Price_in_thousands_avg"].to_numpy()
max_s = np.max(s)
min_s = np.min(s)
scaled = (s - min_s) / (max_s - min_s)

df_2_scaled = df_2
df_2_scaled["Price_in_thousands_avg_scaled"] = scaled
df_2_scaled


# - [x] Randomly split the data into training and test sets using the methods of existing libraries.

# In[15]:


df_2_train = df_2_scaled.sample(frac=0.8, random_state=25)
df_2_test = df_2.drop(df_2_train.index)


# - [x] Train the model on the training set using the **gradient_descent(X, y)** function. Assess model quality on training and test sets using **MSE**, **RMSE** and $R^2$.
# 
# - [x] (+1 point) In all your implementations, add the ability to adjust the necessary hyperparameters, and in the process of training **all** models, select the optimal values ​​of these hyperparameters.

# In[16]:


x_train = np.array([df_2_train["Engine_size"]]).T
y_train = np.array([df_2_train["Price_in_thousands_avg_scaled"]]).T

w_train = gradient_descent(x_train, y_train, err_tol=0.0001)
y_pred_train = np.append(np.ones((x_train.shape[0], 1)), x_train, axis=1) @ w_train


# In[17]:


plt.scatter(x_train, y_train,  color='black')
plt.plot(x_train, y_pred_train, color='blue', linewidth=3)
plt.show()


# In[18]:


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

# In[19]:


regr = Ridge(alpha=0.001, max_iter=1500, tol=0.01)
regr.fit(x_train, y_train)
y_pred_lib = regr.predict(x_train)

plt.scatter(x_train, y_train,  color='black')
plt.plot(x_train, y_pred_train, color='blue', linewidth=3)
plt.plot(x_train, y_pred_lib, color='red', linewidth=3)
plt.show()


# In[20]:


x_test_lib = np.array([df_2_test["Engine_size"]]).T
y_pred_test_lib = regr.predict(x_test_lib)

error_mse_lib = np.sum(((y_pred_test_lib - y_test) ** 2), axis=0) / n_test
error_rmse_lib = np.sqrt(error_mse_lib)
error_r2_lib = 1 - (error_mse_lib * n_test) / np.sum(((y_pred_test_lib - y_test.mean()) ** 2), axis=0)

print(error_mse_lib, error_rmse_lib, error_r2_lib)


# ## Code added on 24.12.2022

# - [x] Repeat the same, but using cross-validation.
# - [x] Create a table, with rows (mse-train, mse-test, rmse-train, rmse-test, r2-train, r2-test) and columns (Fold1, Fold2, ..., Foldk, E, STD) , where k --- number of folds in cross-validation, E --- mat. expectation and STD --- standard deviation. To conclude.

# In[21]:


le = LabelEncoder()
vehicle_types = le.fit_transform(df["Vehicle_type"])
df["Vehicle_type_cat"] = vehicle_types
df


# In[22]:


df_num = df[df.select_dtypes(include=np.number).columns.tolist()]
df_num = df_num.dropna(axis=0)
df_num


# In[23]:


s = df_num["Price_in_thousands"].to_numpy()
max_s = np.max(s)
min_s = np.min(s)
scaled = (s - min_s) / (max_s - min_s)

df_num_scaled = df_num
df_num_scaled["Price_in_thousands_scaled"] = scaled
df_num_scaled = df_num_scaled.drop(["Price_in_thousands"], axis=1)
df_num_scaled


# In[24]:


def get_mse(y_pred, y):
    return np.sum(((y_pred - y) ** 2), axis=0) / y.shape[0]

def get_rmse(y_pred, y):
    return np.sqrt(np.sum(((y_pred - y) ** 2), axis=0) / y.shape[0])

def get_r2(y_pred, y):
    return 1 - (np.sum(((y_pred - y) ** 2), axis=0)) / np.sum(((y_pred - y.mean()) ** 2), axis=0)


# In[25]:


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

comp_data = {}

def training(train, test, fold):
    print("Training for fold no. {0} has started...".format(fold))
    comp_col = {}
    
    x_train = np.array([train["Price_in_thousands_scaled"]]).T
    y_train = np.array([train["Vehicle_type_cat"]]).T
    x_test = np.array([test["Price_in_thousands_scaled"]]).T
    y_test = np.array([test["Vehicle_type_cat"]]).T
    
    w_train = gradient_descent(x_train, y_train, err_tol=0.0001)
    y_pred_train = np.append(np.ones((x_train.shape[0], 1)), x_train, axis=1) @ w_train
    w_test = gradient_descent(x_test, y_test, err_tol=0.0001)
    y_pred_test = np.append(np.ones((x_test.shape[0], 1)), x_test, axis=1) @ w_test
    
    comp_col["mse-train"] = get_mse(y_pred_train, y_train)[0]
    comp_col["mse-test"] = get_mse(y_pred_test, y_test)[0]
    comp_col["rmse-train"] = get_rmse(y_pred_train, y_train)[0]
    comp_col["rmse-test"] = get_rmse(y_pred_test, y_test)[0]
    comp_col["r2-train"] = get_r2(y_pred_train, y_train)[0]
    comp_col["r2-test"] = get_r2(y_pred_test, y_test)[0]
    
    comp_data[str(fold)] = comp_col
    
fold = 1
x = df_num_scaled.drop(["Vehicle_type_cat"], axis=1)
y = df_num_scaled["Vehicle_type_cat"]

for train_index, test_index in cv.split(x, y):
    train = df_num_scaled.iloc[train_index,:]
    test = df_num_scaled.iloc[test_index,:]
    training(train, test, fold)
    fold += 1


# In[26]:


df_comp = pd.DataFrame.from_dict(comp_data)
df_comp


# In[27]:


df_comp["STD"] = df_comp.std(axis=1)
df_comp["E"] = df_comp.drop("STD", axis=1).mean(axis=1)
df_comp


# In[28]:


x = df_num_scaled.drop(["Vehicle_type_cat"], axis=1)
y = df_num_scaled["Vehicle_type_cat"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

params = {"alpha":[1, 10]}
gs_ridge = GridSearchCV(Ridge(), params, scoring="neg_mean_squared_error", cv=cv)

gs_ridge.fit(x_train, y_train)
y_pred_train_lib = gs_ridge.predict(x_train)
get_mse(y_pred_train_lib, y_train)

