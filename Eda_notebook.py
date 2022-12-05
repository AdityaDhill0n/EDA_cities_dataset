#!/usr/bin/env python
# coding: utf-8

# # This notebook deals with Explotary data analysis for the cities dataset of India 

# # Importing the primary Libraries

# In[23]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import warnings
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[2]:


df = pd.read_csv("indian_cities.csv")


# In[3]:


df


# ## Creating a data dictionary for better understanding 
# 
# - Rank - Population rank of the city
# 
# - City - Name of the city
# - State - State of the city 
# - Population - This is the overall population of the main city, this does not include any      population of the suburbs. The data type is continous, and ratio
# - Metro_population - This is the overall population of the city, alongside the metro population. The data type is continous and ratio
# - Sexratio - this is the female to male ratio of the city with representation as N number of females per 1000 males 
# - Literacy - This is the percentage of overall population of city, with basic ability to read and write
# 

# In[4]:


df.info()


# ### Based on the above information we understand there are not missing values 
# 
# ### Rank is described as int64, but since it is rank data it should be considered as object 
# 
# 

# # First we will convert rank to the correct datatype 
# 
# # After the type conversion, we will start by looking at the 4 business moment decisions
# 
# - Measures of central tendency 
# - Measures of dispersion
# - Skewness 
# - Kurtosis 
# 
# # The 4 moments will help in gaining better understanding of the data 

# In[13]:


df['Rank'] = df['Rank'].astype('str')


# In[14]:


df.info()


# In[5]:


# First moment business decision - mean, median and mode


# In[15]:


mean_df = df.mean()
median_df = df.median()
mode_df = df.mode()


# In[20]:


print(mean_df)
print(' ')
print(median_df)
print(' ')
print(mode_df)


# In[21]:


## second business moment - Variance, Standard Deviation and Range


# In[25]:


var_df = df.var()
std_df = df.std()


# In[27]:


print(var_df)
print(' ')
print(std_df)


# In[28]:


## Third Business moment - skewness 


# In[29]:


skewness = df.skew()
print(skewness)


# In[31]:


## Fourth Business moment - Kurtosis 
Kurtosis = df.kurt()
print(Kurtosis)


# In[32]:


# Descriptive statistics 

df.describe()


# # Univariate Analysis using scatter plot

# In[36]:


# Scatter plots for all numerical columns 

for i in df: 
    
    if (df[i].dtypes == np.int64) or (df[i].dtypes == np.float64):
        plt.figure()
        plt.scatter(df.index, df[i])


# # Now lets try to understand the relationship between sexratio, literacy per state

# In[49]:


sns.jointplot(x = 'Literacy', y = 'Sexratio', data = df, hue = 'State', height = 20, ratio = 20 )


# # Histogram and kernel density plot/

# In[50]:


for i in df: 
    
    if (df[i].dtypes == np.int64) or (df[i].dtypes == np.float64):
        plt.figure()
        sns.distplot(df[i])


# # Checking for outliers 

# In[51]:


for i in df: 
    
    if (df[i].dtypes == np.int64) or (df[i].dtypes == np.float64):
        plt.figure()
        sns.boxplot(df[i])


# # Almost all columns have outliers
# 
# # Reducing the range of the columns, by using Standardization and then rechecking for outliers 

# In[52]:


from sklearn.preprocessing import StandardScaler

Sc =  StandardScaler()


# In[53]:


num_values = df[['Population', 'Metro_Population', 'Sexratio', 'Literacy']]


# In[55]:


num_values = Sc.fit_transform(num_values)


# In[56]:


num_values


# In[57]:


num_values = pd.DataFrame(num_values, columns=['Population', 'Metro_Population', 'Sexratio', 'Literacy'])


# In[58]:


for i in num_values:
    plt.figure()
    sns.boxplot(num_values[i])


# # EDA for categorical values

# In[61]:


plt.figure(figsize=(12,12))
sns.countplot(df['State'])
plt.xticks(rotation = 50)
plt.show


# In[ ]:




