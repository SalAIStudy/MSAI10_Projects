"""
Module 1 - Python Essentials
Assignment 1
Student: Salma Areef Syed
Generated: 2025-10-30
This script was exported from the cleaned notebook.
"""

#!/usr/bin/env python
# coding: utf-8

# # Module 1 - Python Essentials
# **Assignment:** 1
# 
# **Student:** Salma Areef Syed
# 
# **Cleaned:** This notebook was cleaned and reformatted for submission.
# 
# **Date cleaned:** 2025-10-30
# 
# ## Objective
# 
# - (Add a short objective description here.)
# 
# ## Contents
# 
# 1. Problem statement
# 2. Code cells (cleaned)
# 3. Results and conclusions
# 
# 

# In[42]:


# 1. Create a DataFrame with Random Data
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randint(1, 101, size=(1000, 5)), columns=list('ABCDE'))
df.head()


# In[43]:


# 2. Check for Missing Values
df.isnull().sum()


# In[44]:


# 3. Fill Missing Values with Mean
df.fillna(df.mean(), inplace=True)
df


# In[45]:


# 4. Add a New Column as Sum of Two Columns
df['F'] = df['A'] + df['B']
df



# In[46]:


# 5. Rename Columns
df.rename(columns={'A': 'Alpha', 'B': 'Beta'}, inplace=True)
df


# In[47]:


# 6. Replace Specific Values
#df['C'].replace(5, 50, inplace=True)
df['C'] = df['C'].replace(5, 50)
df


# In[48]:


# 7. Drop Rows with Any NaN Values
df.dropna(inplace=True)
df


# In[49]:


# 8. Sort DataFrame by a Column
df.sort_values(by='D', ascending=False, inplace=True)
df


# In[50]:


# 9. Filter Rows Based on Condition
df[df['Alpha'] > 50]
df


# In[51]:


# 10. Select Specific Columns
df[['Alpha', 'C']]


# In[52]:


# 11. Group by and Sum
df.groupby('Beta').sum()
df


# In[53]:


#Question 12. Count Unique Values per Group
#df.groupby('B')['C'].nunique()
df.groupby('Beta')['C'].nunique()


# In[54]:


# 13. Maximum Value per Group
df.groupby('Alpha')['D'].max()


# In[55]:


# 14. Apply Custom Function to Groups
df.rename(columns={'Alpha': 'A', 'Beta': 'B'}, inplace=True)
df.groupby('B')['C'].apply(lambda x: x.max() - x.min())


# In[56]:


# 15. Compute Rolling Mean

df['D_rolling'] = df['D'].rolling(window=3).mean()
df['D_rolling']


# In[57]:


# 16. Merge Two DataFrames on a Column
df1 = pd.DataFrame({'ID': [1, 2, 3], 'A': [10, 20, 30]})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'B': [40, 50, 60]})
df_merged = pd.merge(df1, df2, on='ID', how='inner')
df_merged


# In[58]:


# 17. Concatenate DataFrames Row-Wise
df_concat = pd.concat([df1, df2], axis=0)
df_concat


# In[59]:


#18. Find and Remove Duplicates
df.drop_duplicates(inplace=True)
df


# In[60]:


# 19. Pivot Table Creation
df.pivot_table(values='D', index='A', columns='B', aggfunc='mean')


# In[61]:


#20. Extract Year from Date Column
#print(df.columns.tolist())
#df['Year'] = pd.to_datetime(df['Date']).dt.year

import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Date': ['2024-01-15', '2024-05-22', '2025-03-09']
}

df1 = pd.DataFrame(data)

# Convert 'Date' column to datetime
df1['Date'] = pd.to_datetime(df1['Date'])

# Extract the year from the 'Date' column
df1['Year'] = df1['Date'].dt.year

print(df1)


# In[68]:


# 21. Finding Outliers Using IQR
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randint(1, 101, size=(1000, 5)), columns=list('ABCDE'))
df.head()

df.loc[0, 'A'] = 999   # Extreme high value
df.loc[1, 'A'] = -100  # Extreme low value

Q1 = df['A'].quantile(0.25)
Q3 = df['A'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['A'] < (Q1 - 1.5 * IQR)) | (df['A'] > (Q3 + 1.5 * IQR))]
print("Number of outliers:", len(outliers))
outliers


# In[71]:


# 22. Create a Multi-Index DataFrame
#df.set_index(['A', 'B'], inplace=True)
#df
import pandas as pd
import numpy as np

# Recreate sample DataFrame with 'A' and 'B' columns
df = pd.DataFrame(np.random.randint(1, 101, size=(10, 5)), columns=list('ABCDE'))

# Now this will work
df.set_index(['A', 'B'], inplace=True)
print(df)


# In[76]:


# 23. Resample Time Series Data
#df.set_index('Date').resample('M').mean()
import pandas as pd

# Create a sample DataFrame
df1_numeric = df1.select_dtypes(include='number')
df1_numeric['Date'] = df1['Date']
df1_numeric.set_index('Date').resample('ME').mean()

df1.set_index('Date').resample('ME').mean(numeric_only=True)


# In[79]:


#24. Normalize a Column
df['C_norm'] = (df['C'] - df['C'].min()) / (df['C'].max() - df['C'].min())
df


# In[89]:


# 25. Encode Categorical Variables
#df = pd.get_dummies(df, columns=['Category'])
import pandas as pd
import numpy as np

# Recreate sample DataFrame with 'A' and 'B' columns
df = pd.DataFrame(np.random.randint(1, 101, size=(10, 5)), columns=list('ABCDE'))
df = pd.get_dummies(df, columns=['C'])
df



# In[90]:


#26. Finding Correlation Between Columns
df.corr()


# In[93]:


#27. Apply Custom Function to a Column
# Recreate sample DataFrame with 'A' and 'B' columns
df = pd.DataFrame(np.random.randint(1, 101, size=(10, 5)), columns=list('ABCDE'))
df['C_transformed'] = df['C'].apply(lambda x: x * 2 if x > 50 else x / 2)
df


# In[97]:


# 28. Convert JSON Data to DataFrame
#import json
#data = '{"A": [1, 2, 3], "B": [4, 5, 6]}'
#df = pd.read_json(data)

from io import StringIO
import pandas as pd

data = '{"A": [1, 2, 3], "B": [4, 5, 6]}'
df = pd.read_json(StringIO(data))
df


# In[99]:


# 29. Detect and Handle Skewed Data
df['A_log'] = np.log1p(df['A'])
df


# In[102]:


# 30. Split a Column into Multiple Columns
#df[['First', 'Second']] = df['Full Name'].str.split(' ', expand=True)

df2['Full Name'] = ['John Doe', 'Jane Smith', 'Alice Brown']  # Example data
df2[['First', 'Second']] = df2['Full Name'].str.split(' ', expand=True)

df2


# In[104]:


# 31. Convert Dictionary into DataFrame
data = {'A': [1, 2], 'B': [3, 4]}
df = pd.DataFrame(data)
df


# In[105]:


#32. Export DataFrame to CSV Without Index
df.to_csv('output.csv', index=False)


# In[108]:


#33. Read Large CSV in Chunks

def process(chunk):
    # Your logic for processing the chunk goes here
    print(chunk.head())  # This is just an example; you can modify it as needed.
for chunk in pd.read_csv('bank-additional-full.csv', chunksize=1000):
    process(chunk)


# In[109]:


# 34. Find First Non-NaN Value in Each Column
df.apply(lambda x: x.dropna().iloc[0])


# In[112]:


# 35. Use map() for Element-wise Transformations
df = pd.DataFrame(np.random.randint(1, 101, size=(10, 5)), columns=list('ABCDE'))
df['D'] = df['D'].map(lambda x: x**2)
df


# In[113]:


# 36. Find Top 5 Frequent Values in a Column
df['C'].value_counts().head(5)


# In[115]:


#37. Calculate Running Total (Cumulative Sum)
df['Cumulative_Sum'] = df['A'].cumsum()
df


# In[122]:


# 38. Apply a Weighted Mean Function
#df.groupby('B').apply(lambda x: np.average(x['A'], weights=x['C']))
import pandas as pd
import numpy as np

# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['X', 'X', 'Y', 'Y', 'Z'],
    'C': [0.1, 0.2, 0.3, 0.4, 0.5]
})

# Use agg() to calculate the weighted average
result = df.groupby('B').agg(
    WeightedAvg=('A', lambda x: np.average(x, weights=df.loc[x.index, 'C']))
)

print(result)


# In[124]:


# 39. Fill Missing Values Using Forward Fill
#df.fillna(method='ffill', inplace=True)
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'A': [1, None, 3, None, 5],
    'B': [None, 2, None, 4, None]
})

# Using ffill() instead of fillna(method='ffill')
df.ffill(inplace=True)

print(df)



# In[125]:


# 40. Grouping by Multiple Columns and Applying Multiple Aggregations
#df.groupby(['A', 'B']).agg({'C': ['mean', 'sum'], 'D': ['max', 'min']})
#-	... (Expanding up to 100 problems)

import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'A': ['X', 'X', 'Y', 'Y', 'X', 'Y'],
    'B': [1, 2, 1, 2, 1, 2],
    'C': [10, 20, 30, 40, 50, 60],
    'D': [5, 10, 15, 20, 25, 30]
})

# Group by columns 'A' and 'B', then apply multiple aggregation functions
result = df.groupby(['A', 'B']).agg({
    'C': ['mean', 'sum'],
    'D': ['max', 'min']
})

print(result)



# In[127]:


# 41. Creating a Custom Aggregation Function for GroupBy
#df.groupby('B').agg(lambda x: x.median() - x.mean())

import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'A': [10, 20, 30, 40, 50, 60],
    'B': ['X', 'X', 'Y', 'Y', 'X', 'Y'],
    'C': [5, 10, 15, 20, 25, 30]
})

# Custom aggregation function: Difference between median and mean for column 'A'
result = df.groupby('B')['A'].agg(lambda x: x.median() - x.mean())

print(result)



# In[129]:


# 42. Applying a Custom Sorting Order on a Column
#custom_order = {'Low': 1, 'Medium': 2, 'High': 3}
#df['Priority'] = df['Priority'].map(custom_order)
#df.sort_values(by='Priority', inplace=True)

import pandas as pd

# Sample DataFrame with no 'Priority' column initially
df = pd.DataFrame({
    'Task': ['Task 1', 'Task 2', 'Task 3', 'Task 4'],
    'Status': ['Completed', 'In Progress', 'Pending', 'Completed']
})

# Define custom sorting order
custom_order = {'Low': 1, 'Medium': 2, 'High': 3}

# Check if 'Priority' exists in the DataFrame before applying the mapping
if 'Priority' in df.columns:
    df['Priority'] = df['Priority'].map(custom_order)
else:
    print("'Priority' column is missing from the DataFrame")

# If you want to create a 'Priority' column (with some default values), you can do it like this:
df['Priority'] = ['Low', 'Medium', 'High', 'Low']  # Example default values

# Now, map the custom order and sort the DataFrame
df['Priority'] = df['Priority'].map(custom_order)

# Sort by 'Priority' column
df.sort_values(by='Priority', inplace=True)

print(df)



# In[131]:


#43. Finding the Largest 3 Values in Each Column
#df.apply(lambda x: x.nlargest(3).values)
import pandas as pd

# Sample DataFrame with numeric and non-numeric values
df = pd.DataFrame({
    'A': [10, 20, 30, 40],
    'B': [15, 25, 35, 45],
    'C': ['X', 'Y', 'Z', 'W']  # Non-numeric column
})

# Convert all columns to numeric, if possible (ignoring errors for non-numeric columns)
df = df.apply(pd.to_numeric, errors='coerce')

# Now you can safely apply nlargest
top_3_values = df.apply(lambda x: x.nlargest(3).values)

print(top_3_values)



# In[132]:


# 44. Checking if Any Column has Negative Values
df.lt(0).any()


# In[133]:


# 45. Finding the Most Frequent Value in Each Column
df.mode().iloc[0]


# In[134]:


# 46. Detecting Duplicate Rows in a DataFrame
df.duplicated().sum()


# In[138]:


# 47. Extracting Hour from a Timestamp
#df3['Hour'] = pd.to_datetime(df3['Timestamp']).dt.hour
import pandas as pd

# Sample DataFrame df3 with a Timestamp column
df3 = pd.DataFrame({
    'Timestamp': ['2025-05-09 08:30:00', '2025-05-09 12:45:00', '2025-05-09 17:00:00']
})

# Convert 'Timestamp' column to datetime and extract the hour
df3['Hour'] = pd.to_datetime(df3['Timestamp']).dt.hour

print(df3)



# In[140]:


# 48. Finding All Rows with a Specific Pattern in a Column
#df[df['Name'].str.contains('John', case=False, na=False)]
import pandas as pd

# Sample DataFrame df4 with a 'Name' column
df4 = pd.DataFrame({
    'Name': ['John Doe', 'Jane Smith', 'Johnny Depp', 'Mike Johnson', 'Sarah Lee'],
    'Age': [28, 34, 45, 22, 30]
})

# Find all rows where 'Name' contains the pattern 'John', case-insensitive
result = df4[df4['Name'].str.contains('John', case=False, na=False)]

print(result)



# In[141]:


# 49. Converting a Column with Strings to Numeric Values
#df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

import pandas as pd

# Sample DataFrame with a 'Amount' column containing strings
df = pd.DataFrame({
    'Amount': ['100', '200', '300', 'invalid', '500']
})

# Convert 'Amount' column to numeric values, replacing invalid entries with NaN
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Print the result
print(df)



# In[145]:


# 50. Reshaping DataFrame from Wide to Long Format
#df_long = df.melt(id_vars=['ID'], var_name='Feature', value_name='Value')rmat1)
# Melting the DataFrame
import pandas as pd

# Creating the sample dataset
data = {
    'ID': [1, 2, 3],
    'Name': ['John', 'Jane', 'Alice'],
    'Age': [28, 34, 29],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Reshaping the DataFrame from wide to long format
df_long = df.melt(id_vars=['ID'], var_name='Feature', value_name='Value')

# Display the reshaped DataFrame
print(df_long)



