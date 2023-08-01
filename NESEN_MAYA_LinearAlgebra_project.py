#!/usr/bin/env python
# coding: utf-8

# Please follow the instructions carefully. Write all your code in a `Code` cell, and your explanations in a `Markdown` cell. Make sure that your code compiles correctly either by selecting a given cell and clicking the `Run` button, or by hitting `shift`+`enter` or `shift`+`return`.

# ### 1. Import `numpy`, `numpy.linalg`, `matplotlib.pyplot`, and `pandas`. Use the appropriate aliases when importing these modules.

# In[77]:


# code for question 1
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd


# ### 2. Load the data from the file named `data_stock_returns.csv` into a `DatFrame` called `returns`. The file `data_stock_returns.csv` contains daily returns of a number of stocks selected from the S&P 500 universe. The rows of the csv file represent the returns over a number of days, and the columns represent individual stocks labeled by their NYSE ticker symbol, e.g., Apple stock is labeled `AAPL`.

# In[78]:


# code for question 2
returns = pd.read_csv('data_stock_returns.csv')


# ### 3. View the `head` of the `returns` `DataFrame`

# In[79]:


# code for question 3
returns.head()


# ### 4. View the `tail` of the `returns` `DataFrame`

# In[80]:


# code for question 4
returns.tail()


# ### 5. How many stocks are in this `DataFrame`?

# In[81]:


# code for question 5
len(returns.columns)-1
#We subtract 1 for the column with the dates


# **ANSWER FOR QUESTION 5**: There are 488 stocks in the returns dataframe.

# ### 6. Over how many days are these stock returns reported?

# In[82]:


# code for question 6
len(returns.Date)


# **ANSWER FOR QUESTION 6**: There are 252 dates reported.

# ### 7. Extract the returns of the Amazon stock only, which has a ticker symbol `AMZN`. Save it in a `Series` called `amzn_returns`.

# In[83]:


# code for question 7
amzn_returns = np.array(returns["AMZN"])
amzn_returns


# ### 8. Plot the Amazon stock returns extracted in the above cell. 

# In[84]:


# code for question 8
plt.plot(range(1,253), amzn_returns)


# ### 9. Plot the cumulative sum of the Amazon stock returns using the method `.cumsum()` which acts directly on the `amzn_returns` `Series`.

# In[85]:


# code for question 9
import matplotlib.pyplot as plt
plt.plot(amzn_returns.cumsum())


# In[86]:


# the module below will allow us to perform linear regression
import statsmodels.api as sm


# The function `lin_reg(x,y)` given below performs ordinary least squares (OLS) linear regression using `sm.OLS` from the `statsmodels.api` module.
# 
# The code enclosed in `''' '''` is the docstring of the function `lin_reg`.
# 
# `x` in the `lin_reg` function is a matrix that contains the regressors, and `y` represents the vector containing the dependent variable. Note that `x` might contain one vector or multiple vectors. In the case that `x` contains one vector $x$, the regression gives:
# 
# $$ y = \beta_0 + \beta_1 x $$
# 
# In the case that `x` contains multiple vectors $x_1, \dots, x_k$, the regression becomes:
# 
# $$ y = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k$$
# 
# The $\beta$'s are the regression coefficients obtained using least squares. Note that `sm.add_constant` is used in the function below to make `x` look like the matrix $A$ we use in least squares, whose first column contains all ones.

# In[87]:


def lin_reg(x, y):
    '''
    ordinary linear regression using least-squares
    
    Parameters
    ----------  
    x: regressors (numpy array)
    y: dependent variable (numpy array)
    
    Returns
    -------
    coefficients: regression coefficients (pandas Series)
    residuals: regression residuals (numpy array)
    r_squared: correlation coefficient (float)
    
    '''
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    coefficients = model.params
    residuals = model.resid
    r_squared = model.rsquared
    return coefficients, residuals, r_squared


# ### 10. Let's try to use the above function. Extract (as numpy array) the stock returns of:
# 
# - Apple (ticker symbol `AAPL`) and call it `aapl`
# - Intel (ticker symbol `INTC`) and call it `intc`
# - Microsoft (ticker symbol `MSFT`) and call it `msft`
# - IBM (ticker symbol `IBM`) and call it `ibm`
# 
# ### Let `y` be the Apple stock returns, and `x` be the Intel stock returns. Use the `lin_reg` function defined above to find $y=\beta_0 + \beta_1 x$. 

# In[88]:


# code for question 10
aapl = np.array(returns.AAPL)
intc = np.array(returns.INTC)
msft = np.array(returns.MSFT)
ibm = np.array(returns.IBM)

(coefs, _, rsq) = lin_reg(intc,aapl)
[b0, b1]=coefs


# ### 11. Plot the cumulative sum of the Apple returns prediction from least squares on top of the cumulative sum of the actual Apple returns. How well do the Intel stock returns describe the Apple stock returns? Answer this question using a quantitative measure of choice to describe how well the regression describes the actual data (you should research what is standard procedure in answering these types of questions).

# In[89]:


# code for question 11
predicted = 0.00195633 + 0.53526326*intc

plt.plot(aapl.cumsum(), label="actual")
plt.plot(predicted.cumsum(), label="predicted")
plt.legend()
plt.show()


# In[90]:


rsq


# **ANSWER FOR QUESTION 11**: The r-squared value, or the amount of variance in the data, shows that about 30% of Apple's stock returns can be explained by Intel's stock returns.

# ### 12. Now, let `y` be the Apple stock returns, and `x` be the Intel, Microsoft, and IBM stock returns. Use the `lin_reg` function defined above to find $y=\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$, where $x_1$ represents Intel returns, $x_2$ represents Microsoft returns, and $x_3$ represents IBM returns. 

# In[91]:


# code for question 12
y = aapl
x = returns[['INTC', 'MSFT', 'IBM']]
(coefs, _, rsq)= lin_reg(x,y)


# ### 13. Plot the cumulative sum of the Apple returns prediction from least squares on top of the cumulative sum of actual Apple returns. How well do the Intel, Microsoft, and IBM stock returns describe the Apple stock returns? Answer this question using a quantitative measure of choice to describe how well the regression describes the actual data (you should research what is standard procedure in answering these types of questions).

# In[92]:


# code for question 13
plt.plot(aapl.cumsum(), label = "actual")
predicted2 = 0.001116 + 0.284141*intc + 0.543723*msft + 0.150586*ibm

plt.plot(predicted2.cumsum(), label = 'predicted')
plt.legend()
plt.show()


# In[93]:


rsq


# **ANSWER FOR QUESTION 13**: The r-sqared value shows that about 47% of Apple's stock returns can be explained by INTC, MSFT, and IBM stock returns.

# The file `SPY.csv` contains the prices of SPDR S&P 500 ETF Trust. This Exchange Traded Fund (ETF) contains a collection of assets currently present in the S&P 500 index. 
# 
# ### 14. Load `SPY.csv` into a DataFrame called `spy_prices` using the `read_csv` method in `pandas`. Make sure to make the 'Date' column to be your index column. To do that, read the docstring for `read_csv`. 

# In[94]:


# code for question 14
spy_prices = pd.read_csv('SPY.csv', index_col = 'Date')


# ### 15. Once you have downloaded the file into the `DataFrame`, observe all the available prices and dates. Show the head and tail of the `DataFrame`, and then answer the following questions:
# 
# (a) Which types of prices are reported?
# 
# (b) From which date to which date are these prices reported (in the entire DataFrame)?

# In[95]:


# code for question 15
spy_prices.head()
spy_prices.tail()


# **ANSWER FOR QUESTION 15**: (double click here and type your answer)
# 
# (a) High, low, open, close, adj close
# 
# (b) From 2020-05-26 to 2020-06-01

# ### 16. Retain only the Adjusted Close price in the `spy_prices` `DataFrame`. Call the resulting `Series` `spy_adjclose`.

# In[96]:


# code for question 16
spy_adjclose = spy_prices['Adj Close']
spy_adjclose


# ### 17. Now, using the `pct_change` method in `pandas`, compute the returns on the Adjusted Close prices of SPY, and only retain the returns from '2019-01-01' to '2020-01-01'. Call the `Series` obtained `spy_returns`.

# In[97]:


# code for question 17
spy_returns = spy_adjclose.pct_change().loc['2019-01-01':'2020-01-01']
spy_returns


# ### 18. Perform SVD on `returns` data that contain assets from the S&P 500, and save the left singular vectors in an array called `U`, the singular values in an array called `sing_vals`, and the right singular vectros in an array called `V`. Retain the left singular vector corresponding to the largest singular value and call is `u_sigma1`.

# In[98]:


# code for question 18
U, sing_vals, V = la.svd(returns.iloc[:,1:])
u_sigmal = U[:,[0]]
u_sigmal


# ### 19. `u_sigma1` is thought to track the market. To test that, we will perform a regression of `spy_returns` against this first left singular vector by letting `y=spy_returns` and `x=u_sigma1` and computing
# 
# ### $$ y = \beta_0 + \beta_1 x$$
# ### using least squares regression.

# In[99]:


# code for question 19
(coefs,_, rsq,)= lin_reg(u_sigmal, spy_returns)


# ### 20. Plot the cumulative sum of the result from the regression on top of the cumulative sum of `spy_returns`. What do you notice? Answer this question using a quantitative measure of choice to describe how well the regression describes the actual data (you should research what is standard procedure in answering these types of questions).

# In[100]:


# code for question 20
plt.plot(spy_returns.cumsum(), label = 'actual')
predicted3 = 0.000346 - 0.118992*u_sigmal

plt.plot(predicted3.cumsum(), label = 'predicted')
plt.legend()
plt.show()


# In[101]:


rsq


# **ANSWER FOR QUESTION 20**: The r-squared values shows that about 90% of the variance in the SPY returns data can be explained by the largest singular value.

# # Congratulations! You have just implemented your first PCA regression to describe the returns on the S&P 500. Now let's shift gears a little and see how singular values and singular vectors of returns matrices are connected to eigenvalues and eigenvectors of correlation matrices.

# ### 21. Compute the standardized returns of the S&P 500 and save you answer in a `DataFrame` called `sp500_standardized_returns`.

# In[102]:


# code for question 21
sp500_standardized_returns = (returns-returns.mean())/returns.std()


# ### 22. Perform SVD on `sp500_standardized_returns` data and save the singular values in an array called `singvals_standardized_returns`.

# In[103]:


# code for question 22
sp500_std_nodate= sp500_standardized_returns.drop('Date', axis=1)
u, singvals_standardized_returns, v = la.svd(sp500_std_nodate)
v


# ### 23. Compute the correlation matrix of the S&P 500 returns and save the result as a `DataFrame` called `sp500_corr`. It is easiest to use the built in `pandas` method to compute correlation matrices. 

# In[104]:


# code for question 23
sp500_corr = sp500_standardized_returns.corr()
sp500_corr


# ### 24. Compute the eigenvalues and eigenvectors of the correlation matrix. Save the results in arrays called `eigvals_corr` and `eigvecs_corr`. How are the eigenvalues of `sp500_corr` related to the singular values of `sp500_standardized_returns`? You should derive this result theoretically.

# In[105]:


# code for question 24
eigvals_corr, eigvecs_corr = la.eig(sp500_corr)
eigvecs_corr


# **ANSWER FOR QUESTION 24**: The eigenvalues are the square root of the singular values.

# ### 25. What is the sum of the eigenvalues? Justify why the sum turns out to be what it is.

# In[106]:


# code for question 25
sum(eigvals_corr)


# **ANSWER FOR QUESTION 25**: The sum of the eigenvalues is the trace of the correlation matrix. It is 488 because the diagonal of the correlation matrix is only 1s and there are 488 rows/columns.

# ### 26. How are the eigenvectors of the correlation matrix `eigvecs_corr` connected to the left singular vectors of the standardized returns matrix in Question 22? Here you are expected to derive the relation theoretically.

# **ANSWER FOR QUESTION 26**: The eigenvectors of the correlation matrix are the transpose of the left singular vectors of the standardized returns matrix.
