"""
Module 2 - Data Processing for AI
Merged notebooks: Day2 PandasSQL & Day2to4 SQL
Student: Salma Areef Syed
Generated: 2025-10-30
DB-related code has been commented out for safe execution.
"""

#!/usr/bin/env python
# coding: utf-8

# # Module 2 - Data Processing for AI
# **Assignments merged:** Pandas & SQL Exercises
# **Student:** Salma Areef Syed
# **Date merged:** 2025-10-30
# 
# ## Objective
# 
# - Demonstrate Pandas data processing and SQL-style operations.
# 
# 

# ## Part A — Pandas & SQL Exercises (Day2)

# In[1]:


#!pip install pandas==2.0.3
get_ipython().system('pip install pandasql')


# In[1]:


#p install pandas==2.0.3
get_ipython().system('pip install pandas --upgrade')


# In[2]:


import numpy as np
import pandas as pd
import pandasql as ps
import seaborn as sns
from pandasql import sqldf


# In[3]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[4]:


df = sns.load_dataset('titanic')
df.head()
df.tail()


# In[12]:


pysqldf = lambda q: sqldf(q, globals())


# In[13]:


q = '''
SELECT * from df
'''
output = pysqldf(q)
output


# In[15]:


# Run 10 different queries for the selected datafram to answer key questions of the data
# 1. Find total number of passengers
q = '''
SELECT COUNT(*) AS total_passengers FROM df
'''
output = pysqldf(q)
output


# In[17]:


# 2. how many passengers survieved in which deck?

q = '''
SELECT
  distinct(deck)
FROM df

'''
output = pysqldf(q)
output


# In[18]:


# 2. how many passengers survived in which deck?

q = '''
SELECT
  deck,
  COUNT(*) AS total_passengers,
  SUM(survived) AS survived_passengers,
  ROUND(SUM(survived)*100.0 / COUNT(*), 2) AS survival_rate_percent
FROM df
WHERE deck IS NOT NULL
GROUP BY deck
ORDER BY deck;
'''
output = pysqldf(q)
output


# In[19]:


#3. Identify Passengers Traveling Alone vs With Family and Compare Survival
q = '''
SELECT
  CASE
    WHEN (sibsp + parch) = 0 THEN 'Alone'
    ELSE 'With Family'
  END AS travel_group,
  COUNT(*) AS total,
  SUM(survived) AS survivors,
  ROUND(AVG(survived) * 100, 2) AS survival_rate
FROM df
GROUP BY travel_group;
'''
output = pysqldf(q)
output


# In[20]:


#4. Average age of survivors vs non-survivors, grouped by gender
q = '''
SELECT
  sex,
  survived,
  ROUND(AVG(age), 2) AS avg_age
FROM df
WHERE age IS NOT NULL
GROUP BY sex, survived;
'''
output = pysqldf(q)
output


# In[24]:


#5. Passengers with fare above the 95th percentile

fare_95 = df['fare'].quantile(0.95)
q = '''
SELECT *
FROM df
WHERE fare > {}
ORDER BY fare DESC;
'''.format(fare_95)
output = pysqldf(q)
output


# In[26]:


#6. How many passengers between age 0 , 1
q = '''
SELECT COUNT(*) AS infant_passenger_count
FROM df
WHERE age >= 0 AND age <= 1;
'''
output = pysqldf(q)
output


# In[29]:


#7. Find Passengers Paying Above the Average Fare in Their Class

q = '''
SELECT *
FROM df t1
WHERE fare > (
  SELECT AVG(fare)
  FROM df t2
  WHERE t2.pclass = t1.pclass
)
ORDER BY fare DESC;
'''
output = pysqldf(q)
output


# In[32]:


#8. Average Age of Survivors vs Non-Survivors who embarked in Southampton town

q = '''
SELECT
  survived,
  ROUND(AVG(age), 2) AS avg_age,
  COUNT(*) AS count_passengers
FROM df
WHERE embarked = 'S' AND age IS NOT NULL
GROUP BY survived;
'''
output = pysqldf(q)
output



# In[40]:


#9. Survival count by gender

q = '''
SELECT sex, survived, COUNT(*) AS count
FROM df
GROUP BY sex, survived;
'''
output = pysqldf(q)
output


# In[47]:


#10.  Survival rate by passenger class
df.columns

q = '''
SELECT class, AVG(survived)*100 AS survival_rate
FROM df
GROUP BY class
'''
output = pysqldf(q)
output


# ## Part B — SQL Assignments (Day2-4)

# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head()
#Print multiple statemetns in same line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
titanic.info()


# In[8]:


titanic.drop(columns = ['who', 'adult_male', 'embark_town', 'alone', 'alive', 'class', 'deck'], inplace = True)
titanic.info()


# In[9]:


titanic['embarked'].mode()


# In[11]:


titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic.info()


# In[12]:


titanic.sample(10)


# In[14]:


for x in titanic.columns:
  if titanic[x].dtype == "object":
    titanic[x]=pd.Categorical(titanic[x]).codes


# In[15]:


titanic.info()


# In[18]:


np.arange(0,1.01,0.05)


# In[19]:


titanic.quantile(np.arange(0,1.01,0.05))

It calculates quantiles (percentiles) for numerical columns in the titanic DataFrame at every 5% interval, from 0% to 100%.


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
titanic.fare.hist()
plt.show()


# In[21]:


print(np.round(titanic.fare.median()),2)
print(np.round(titanic.fare.mean(),2))


# In[22]:


titanic.boxplot(column=['fare'])


# In[24]:


titanic.fare.quantile([0,0.01,0.02,0.05,0.90,0.95,0.99,1.00])


# In[25]:


titanic.fare.describe()


# In[30]:


titanic.fare = np.clip(titanic.fare, titanic.fare.quantile(0.02), titanic.fare.quantile(0.99))
titanic.fare.describe()
titanic.info()


# In[33]:


for x in titanic.columns:
  outlier = titanic[x].quantile([0.01,0.99]).values
  titanic[x] = np.clip(titanic[x],outlier[0],outlier[1])
  titanic.describe()


# In[35]:


titanic.shape


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# In[37]:


x = titanic.drop(['survived'], axis=1, inplace=False)
y = titanic['survived']
x.shape
print('\n')
y.shape


# In[42]:


from sklearn.preprocessing import MinMaxScaler
scld_titanic = MinMaxScaler(feature_range= (0,1))
titanic_transformed = scld_titanic.fit_transform(x)
scld_titanic_df = pd.DataFrame(titanic_transformed, columns=x.columns)
scld_titanic_df.head()
scld_titanic_df.describe()


# In[43]:


from sklearn.preprocessing import StandardScaler
scld_titanic = StandardScaler()
titanic_transformed = scld_titanic.fit_transform(x)
scld_titanic_df = pd.DataFrame(titanic_transformed, columns=x.columns)
scld_titanic_df.head()
np.round(scld_titanic_df.describe(),2)

