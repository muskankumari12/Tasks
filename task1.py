#!/usr/bin/env python
# coding: utf-8

# # GRIP: THE SPARKS FOUNDATION

# # Data Science and Business Analytics

# # Task 1: Prediction Using Supervised ML

# In this task we have to predict the percentage score of a  students
# based on the number of hours studied. The task has two variables
# where the feature is the number of hours studied and the target value
# is the percentage score. This can be solved using simple linear regression.

# In[23]:


# Importing necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[24]:


# Reading data

url = "http://bit.ly/w-data"
data = pd.read_csv(url)


# Exploring Data

# In[8]:


data.head(10) # top ten data


# In[10]:


data.isnull==True # to check whether the data contain null value of not


# In[22]:


data.shape #knowing the shape of data is good to analyze and build model.


# In[23]:


data.info


# In[24]:


data.describe()


# Lets plot the graph to see if there is any relationship between data or not
# 

# In[14]:


#plotting the distribution of scores

data.plot.scatter(x = "Hours", y = "Scores", title = " Scatter plot of Hours and Scores Relationship", color="red");


# From the above graph, we can clearly see that there is positive
# linear relation between the number of hours studied and percentages
# of score

# In[27]:


data.corr() # corr() method check and display how the variable are corelated.


# The next step is divide the data into "attributes" (inputs)
# and "labels" (outputs)

# In[15]:


X = data.iloc[:, :-1].values
Y = data.iloc[:, 1].values


# Now we have our labels and attributes, the next step is to split data into
# training and test sets. We'll do this using Sckikit-Learn's built-in
# train_test_split() method.

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state =0) 


# # Training the Algorithm
# 
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.
# 

# In[17]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[18]:


# Plotting the regression line

line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X,Y, color="green")
plt.plot(X, line);
plt.show()


# # Making Predictions
# 
# Now that we have trained our algorithm, it's time to make some prediction

# In[19]:


print(X_test) # Testing data - In Hours
Y_pred = regressor. predict(X_test) # Predicting


# In[20]:


# Comparing Actual vs Predicted

df = pd.DataFrame({"Actual": Y_test, "Predicted": Y_pred})
df


# In[22]:


# You can also test with your own data
hours = ([[9.25]])
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Evaluating the model
# 
# The final step is to evaluate the performance of algoritnm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.
# 

# In[35]:


from sklearn import metrics
print('Mean Absolute Error:',
      metrics.mean_absolute_error(Y_test, Y_pred))


# In[ ]:





# In[ ]:




