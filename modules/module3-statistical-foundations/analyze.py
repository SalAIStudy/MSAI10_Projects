"""
Module 3 - Statistical Foundations for AI
Merged notebook -> analyze.py
Student: Salma Areef Syed
Generated: 2025-10-30
DB/external connection code is commented out for safe submission.
"""

#!/usr/bin/env python
# coding: utf-8

# # Module 3 - Statistical Foundations for AI
# **Merged notebooks:** 3 notebooks
# **Student:** Salma Areef Syed
# **Date merged:** 2025-10-30
# 
# ## Objective
# 
# - Demonstrate statistical analysis, visualization, and applied examples for AI.
# 
# 

# ## Assignment 01 — Statistical Foundations
# 
# *Original notebook: `Assignment_01_StatisticalFoundation_AI10_SalmaSyed.ipynb`*
# 

# Hypothesis Testing: Parametric vs Non-Parametric Analysis Based on Data Distribution

# In[ ]:


Assignment Objectives:
By completing this assignment, you will be able to:
•	Understand when to use parametric vs non-parametric tests
•	Perform distribution checks (normality tests, plots)
•	Formulate and test null and alternative hypotheses
•	Use at least one parametric and one non-parametric test appropriately
•	Interpret results clearly with visuals and reasoning
Dataset:
 Employee Attrition Dataset – Kaggle
Dataset Features (Selected)
•	Age (numeric)
•	MonthlyIncome (numeric)
•	JobSatisfaction (1–4)
•	Attrition (Yes/No)
•	JobRole (Sales Executive, Research Scientist, etc.)
Problem Statement:
You are an HR data analyst at a large firm. The leadership wants to understand whether:
•	Monthly income and age differ between employees who left vs stayed
•	Job satisfaction differs across job roles
Use statistical hypothesis testing to answer the following:

get_ipython().run_line_magic('pinfo', 'Parametric')
1.	Select two numeric variables (MonthlyIncome, Age) and one categorical grouping variable (Attrition or JobRole).
2.	Check the distribution of the numeric variables using:
•	Histogram / Boxplot
•	Shapiro-Wilk or Kolmogorov-Smirnov test
3.	Decide whether each variable is normally distributed or not.


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro, kstest
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# =========================
# 📂 Step 1: Data Loading
# =========================
print("📥 Loading dataset...")

# Load dataset from CSV
file_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Explanation
print("\n✅ Data successfully loaded.")
print("   → Total records:", df.shape[0])
print("   → Total features:", df.shape[1])

# =========================
# 🧹 Step 2: Data Cleaning
# =========================
print("\n🧹 Cleaning Data...")

# Check for missing values
missing_summary = df.isnull().sum()
missing_columns = missing_summary[missing_summary > 0]

if missing_columns.empty:
    print("   → No missing values found in the dataset.")
else:
    print("   ⚠️ Missing values found in the following columns:")
    print(missing_columns)

# You could add imputation here if needed; dataset has no missing by default

# ==============================
# 📊 Step 3: Basic EDA Summary
# ==============================
numeric_vars = ['MonthlyIncome', 'Age']
print("\n📊 Descriptive Statistics for Numeric Variables:")
print(df[numeric_vars].describe())

# Explanation:
print("\n🧾 Explanation:")
print("   → We are analyzing the distribution of two numeric variables: MonthlyIncome and Age.")
print("   → Summary statistics (count, mean, std, min, max, quartiles) are shown above.")

# ===============================
# 🎯 Step 4: Normality Testing
# ===============================
print("\n🎯 Checking if variables follow a normal distribution.")
print("   H₀: Data follows a normal distribution.")
print("   H₁: Data does not follow a normal distribution.")

summary_results = []

# Loop over numeric variables
for var in numeric_vars:
    print(f"\n🔍 Variable: {var}")

    # Visual Analysis
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(df[var], kde=True, bins=30, color='skyblue')
    plt.title(f'{var} - Histogram + KDE')

    plt.subplot(1, 3, 2)
    sns.boxplot(x=df[var], color='lightgreen')
    plt.title(f'{var} - Boxplot')

    plt.subplot(1, 3, 3)
    stats.probplot(df[var], dist="norm", plot=plt)
    plt.title(f'{var} - Q-Q Plot')

    plt.tight_layout()
    plt.show()

    # Numeric summaries
    skew_val = df[var].skew()
    kurt_val = df[var].kurtosis()
    print(f"📈 Skewness: {skew_val:.2f}")
    print(f"📈 Kurtosis: {kurt_val:.2f}")

    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = shapiro(df[var])
    print(f"\n🧪 Shapiro-Wilk Test → W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print(f"   ✅ Interpretation: Data likely comes from a normal distribution (fail to reject H₀).")
        print(f"   → ✅ Fail to Reject H₀ → Likely Normal")
    else:
        print(f"   ❌ Interpretation: Data likely does NOT come from a normal distribution (reject H₀).")
        print(f"   → ❌ Reject H₀ → Not Normal")

    # Kolmogorov-Smirnov Test
    standardized = (df[var] - df[var].mean()) / df[var].std()
    ks_stat, ks_p = kstest(standardized, 'norm')
    print(f"\n🧪 Kolmogorov-Smirnov Test → D = {ks_stat:.4f}, p = {ks_p:.4f}")
    if ks_p > 0.05:
        print(f"   ✅ Interpretation: Data likely comes from a normal distribution (fail to reject H₀).")
        print(f"   → ✅ Fail to Reject H₀ → Likely Normal")
    else:
        print(f"   ❌ Interpretation: Data likely does NOT come from a normal distribution (reject H₀).")
        print(f"   → ❌ Reject H₀ → Not Normal")

    # Final Decision
    final_decision = "✅ Normally Distributed" if shapiro_p > 0.05 and ks_p > 0.05 else "❌ Not Normally Distributed"
    print(f"\n📌 Final Decision: {final_decision}")

    # Save results
    summary_results.append({
        'Variable': var,
        'Skewness': round(skew_val, 2),
        'Kurtosis': round(kurt_val, 2),
        'Shapiro p-value': round(shapiro_p, 4),
        'KS p-value': round(ks_p, 4),
        'Normally Distributed?': "Yes" if final_decision.startswith("✅") else "No"
    })

# ============================
# 📋 Summary Table of Results
# ============================
summary_df = pd.DataFrame(summary_results)
print("\n📊 Final Normality Summary Table:")
print(summary_df.to_string(index=False))

# ========================
# 🔗 Correlation Heatmap
# ========================
plt.figure(figsize=(5, 4))
corr = df[['MonthlyIncome', 'Age', 'JobSatisfaction']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("🔗 Correlation Heatmap")
plt.show()

# Interpretation of Heatmap
print("\n📌 Heatmap Interpretation:")
for row in corr.index:
    for col in corr.columns:
        if row != col:
            r = corr.loc[row, col]
            direction = "positive" if r > 0 else "negative"
            strength = abs(r)
            if strength >= 0.7:
                desc = "strong"
            elif strength >= 0.4:
                desc = "moderate"
            elif strength >= 0.1:
                desc = "weak"
            else:
                desc = "negligible"
            print(f" - {row} and {col}: {desc} {direction} correlation (r = {r:.2f})")


# PART B — Hypothesis Test (Binary Group)
# Example Hypothesis:
# •	H₀: There is no difference in Monthly Income between employees who left and those who stayed.
# •	H₁: There is a difference in Monthly Income between the two groups.
#  If normal: Use Independent Two-Sample t-test
#  If not normal: Use Mann-Whitney U Test

# In[15]:


from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import pandas as pd

# Extract groups
group_yes = df[df['Attrition'] == 'Yes']['MonthlyIncome'].dropna()
group_no = df[df['Attrition'] == 'No']['MonthlyIncome'].dropna()

# --- Step 1: State the Hypotheses ---
print("\n🎯 Hypothesis:")
print("H₀: There is no difference in Monthly Income between employees who left and those who stayed.")
print("H₁: There is a difference in Monthly Income between the two groups.")

# --- Step 2: Check Normality of Both Groups ---
print("\n📊 Normality Check using Shapiro-Wilk Test:")
normal_yes = shapiro(group_yes)
normal_no = shapiro(group_no)

def interpret_shapiro(label, stat, p):
    verdict = "✅ Likely Normal" if p > 0.05 else "❌ Not Normal"
    print(f"  - Attrition = {label} → W = {stat:.4f}, p = {p:.4f} → {verdict}")

interpret_shapiro('Yes', *normal_yes)
interpret_shapiro('No', *normal_no)

# --- Step 3: Choose the Test Based on Normality ---
print("\n🔍 Test Selection Rationale:")
if normal_yes.pvalue > 0.05 and normal_no.pvalue > 0.05:
    print("✅ Both groups appear to be normally distributed based on p > 0.05.")
    print("→ Using **Independent Two-Sample t-test**, a parametric test suitable when both groups follow normal distribution.")
    stat, p = ttest_ind(group_yes, group_no, equal_var=False)
    test_used = "Independent t-test"
else:
    print("❌ One or both groups do not appear to be normally distributed.")
    print("→ Using **Mann-Whitney U Test**, a non-parametric test that does not assume normality.")
    stat, p = mannwhitneyu(group_yes, group_no, alternative='two-sided')
    test_used = "Mann-Whitney U Test"

# --- Step 4: Report Results ---
print(f"\n🧪 Test Used: {test_used}")
print(f"   → Test Statistic = {stat:.4f}, p-value = {p:.4f}")

# --- Step 5: Conclusion ---
print("\n📌 Final Conclusion:")
if p < 0.05:
    print("❌ Reject H₀: There IS a significant difference in Monthly Income between employees who left and those who stayed.")
else:
    print("✅ Fail to Reject H₀: There is NO significant difference in Monthly Income between the groups.")


# PART C — Hypothesis Test (Multiple Groups)
# Example Hypothesis:
# •	H₀: The median job satisfaction is the same across all job roles.
# •	H₁: At least one job role differs in median job satisfaction.
#  If normal: Use One-way ANOVA
#  If not normal: Use Kruskal-Wallis Test

# In[16]:


from scipy.stats import shapiro, kruskal, f_oneway
import pandas as pd

# Assuming df is already loaded and cleaned
job_roles = df['JobRole'].unique()
group_data = {role: df[df['JobRole'] == role]['JobSatisfaction'] for role in job_roles}

# 1. Check normality within each job role
normality_results = {}
print("📊 Normality Check per Job Role (Shapiro-Wilk):")
for role, scores in group_data.items():
    stat, p = shapiro(scores)
    is_normal = p > 0.05
    symbol = "✅" if is_normal else "❌"
    normality_results[role] = is_normal
    print(f"  - {role}: W = {stat:.4f}, p = {p:.4f} → {symbol} {'Likely Normal' if is_normal else 'Not Normal'}")

# 2. Choose appropriate test
if all(normality_results.values()):
    print("\n✅ All groups passed normality → Using One-way ANOVA")
    test_name = "One-way ANOVA"
    stat, p = f_oneway(*(scores for scores in group_data.values()))
else:
    print("\n❌ Not all groups are normally distributed → Using Kruskal-Wallis Test")
    test_name = "Kruskal-Wallis Test"
    stat, p = kruskal(*(scores for scores in group_data.values()))

# 3. Report results
print(f"\n🧪 Test Used: {test_name}")
print(f"   → Test Statistic = {stat:.4f}, p-value = {p:.4f}")

if p < 0.05:
    print("❌ Reject H₀: There is a significant difference in Job Satisfaction across Job Roles.")
else:
    print("✅ Fail to Reject H₀: No significant difference in Job Satisfaction across Job Roles.")


# Bonus Challenge (Optional)
#  Compare Age across JobRoles using both parametric and non-parametric tests, and explain:
# Which test is more appropriate and why?

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, f_oneway, kruskal


# Filter relevant columns
df = df[['Age', 'JobRole']]

# Preview
print("🔍 Unique Job Roles:", df['JobRole'].nunique())
print("📊 Sample Data:\n", df[['JobRole', 'Age']].head())

# Check group-wise normality using Shapiro-Wilk test
normality_results = {}
print("\n🧪 Shapiro-Wilk Normality Test per JobRole:")
for role in df['JobRole'].unique():
    age_data = df[df['JobRole'] == role]['Age']
    stat, p = shapiro(age_data)
    normality_results[role] = p > 0.05
    status = "✅ Likely Normal" if p > 0.05 else "❌ Not Normal"
    print(f"  - {role:25s} → W = {stat:.4f}, p = {p:.4f} → {status}")

# Decide which test is more appropriate
all_normal = all(normality_results.values())
if all_normal:
    print("\n✅ All job roles passed normality → One-way ANOVA is appropriate.")
else:
    print("\n❌ Not all job roles are normally distributed → Kruskal-Wallis is more appropriate.")

# Prepare data for testing
groups = [df[df['JobRole'] == role]['Age'] for role in df['JobRole'].unique()]

# Run both tests for comparison
print("\n📊 Parametric Test → One-way ANOVA")
anova_stat, anova_p = f_oneway(*groups)
print(f"   F-statistic = {anova_stat:.4f}, p-value = {anova_p:.4f}")
if anova_p < 0.05:
    print("   ❌ Reject H₀: Significant difference in Age across JobRoles (ANOVA)")
else:
    print("   ✅ Fail to Reject H₀: No significant difference in Age (ANOVA)")

print("\n📊 Non-Parametric Test → Kruskal-Wallis")
kw_stat, kw_p = kruskal(*groups)
print(f"   H-statistic = {kw_stat:.4f}, p-value = {kw_p:.4f}")
if kw_p < 0.05:
    print("   ❌ Reject H₀: Significant difference in Age across JobRoles (Kruskal-Wallis)")
else:
    print("   ✅ Fail to Reject H₀: No significant difference in Age (Kruskal-Wallis)")

# Final Recommendation
print("\n📌 Final Recommendation:")
if all_normal and anova_p < 0.05:
    print("✅ One-way ANOVA is statistically appropriate and shows significant difference.")
elif not all_normal and kw_p < 0.05:
    print("✅ Kruskal-Wallis is more appropriate due to non-normality and it shows significance.")
elif not all_normal:
    print("🔎 Although ANOVA was performed, Kruskal-Wallis is preferred because not all groups are normal.")
else:
    print("🟰 Both tests suggest no significant difference. Use Kruskal-Wallis if assumptions are violated.")

# Optional: Boxplot visualization
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='JobRole', y='Age', palette='Set2')
plt.xticks(rotation=45)
plt.title("🎨 Age Distribution across Job Roles")
plt.tight_layout()
plt.show()


# ## Day 1 — Visualization
# 
# *Original notebook: `Module03_Day1_Visualization.ipynb`*
# 

# In[5]:


import pandas as pd
df = pd.read_csv("data/netflix_titles.csv")
df.head()


# In[4]:


df.describe(include="all")


# In[6]:


import pandas as pd

# Load the dataset
df = pd.read_csv('data/netflix_titles.csv')

# Function to convert duration to minutes
def convert_to_minutes(duration, type_):
    if pd.isna(duration):
        return None
    try:
        # Remove leading/trailing whitespace
        duration = duration.strip()

        if 'min' in duration:
            # Movie duration (e.g., "90 min")
            return int(duration.replace('min', '').strip())
        elif 'Season' in duration:
            # TV show duration (e.g., "2 Seasons" or "1 Season")
            num_seasons = int(duration.replace('Seasons', '').replace('Season', '').strip())
            # Assume 10 episodes per season, 45 minutes per episode
            return num_seasons * 10 * 45
        else:
            return None
    except (ValueError, AttributeError):
        return None

# Create new column 'duration_in_minutes'
df['duration_in_minutes'] = df.apply(lambda row: convert_to_minutes(row['duration'], row['type']), axis=1)

# Display the first few rows to verify
df[['title', 'type', 'duration', 'duration_in_minutes']].head(10)


# In[7]:


df.shape


# In[8]:


df.head()


# In[14]:


df.describe(include="all")


# In[15]:


df.describe()


# In[9]:


df.boxplot(column=["duration_in_minutes"])


# In[10]:


px.box(df, y='duration_in_minutes')


# In[17]:


df.drop_duplicates(inplace=True)


# In[18]:


df.shape


# In[19]:


df.isnull().sum()


# In[21]:


#mode of rating
df['rating'].fillna(df['rating'].mode()[0], inplace=True)


# In[22]:


df.isnull().sum()


# In[24]:


#make null value of director as unknown
df['director'].fillna('Unknown', inplace=True)


# In[25]:


df.isnull().sum()


# In[27]:


df.describe()


# In[8]:


df = pd.read_csv("data/StudentsPerformance.csv")
df.head()


# In[9]:


df.boxplot(column=["math score","reading score","writing score"])


# In[4]:


from pandasql import sqldf


# In[3]:


get_ipython().system('pip install pandasql')


# In[14]:


query ="select * from df group by raceethnicity;"
op = sqldf(query)
op


# In[15]:


df = pd.read_csv("data/ds_salaries.csv")
df.head()


# In[16]:


df.boxplot(column=["salary","salary_in_usd"])


# In[17]:


df.shape


# In[24]:


import pandas as pd
import plotly.express as px


# Convert salary column to numeric if not already
df['salary_in_usd'] = pd.to_numeric(df['salary_in_usd'], errors='coerce')

# Drop missing values in relevant columns
df = df.dropna(subset=['salary_in_usd', 'job_title', 'experience_level', 'company_size'])




fig1 = px.box(df,
              x='job_title',
              y='salary_in_usd',
              title='USD Salary Distribution by Job Title',
              labels={'job_title': 'Job Title', 'salary_in_usd': 'Salary (USD)'},
              color='job_title')
fig1.update_layout(xaxis_tickangle=45)
fig1.show()

#Boxplot by Experience Level
fig2 = px.box(df,
              x='experience_level',
              y='salary_in_usd',
              title='USD Salary Distribution by Experience Level',
              labels={'experience_level': 'Experience Level', 'salary_in_usd': 'Salary (USD)'},
              color='experience_level')
fig2.show()

#Boxplot by Company Size
fig3 = px.box(df,
              x='company_size',
              y='salary_in_usd',
              title='USD Salary Distribution by Company Size',
              labels={'company_size': 'Company Size', 'salary_in_usd': 'Salary (USD)'},
              color='company_size')
fig3.show()


# In[21]:


df['job_title'].nunique()


# In[25]:


df = pd.read_csv("data/Titanic-Dataset.csv")
df.head()


# In[34]:


#scatterplot of age and fare colur by survival
px.scatter(df, x='Age', y='Fare', color='Survived')


# In[33]:


#histogram of Age
df['Age'].hist(bins = 20)


# In[36]:


px.scatter(df, x='Sex', y='Fare', color='Survived')


# In[46]:


df = pd.read_csv("data/AirQualityUCI.csv",sep = ";",decimal=',')
df.head()


# In[41]:


df['CO(GT)'].isnull().sum()


# In[42]:


df.shape


# In[43]:


df.isnull().sum()


# In[48]:


df['Date'].nunique()


# In[47]:


df.describe()


# In[49]:


# Calculate the mean of 'CO(GT)' column, excluding NaN [before]
mean_co = df['CO(GT)'].mean()

print(f"Mean of CO(GT): {mean_co}")


# In[50]:


# Apply linear interpolation on 'CO(GT)' column
df['CO(GT)'] = df['CO(GT)'].interpolate(method='linear')

# Check if any NaNs remain and view the filled column
print(f"Missing values after interpolation: {df['CO(GT)'].isnull().sum()}")
print(df['CO(GT)'])  # View first 10 rows of interpolated column


# In[51]:


# Calculate the mean of 'CO(GT)' column, excluding NaN [after]
mean_co = df['CO(GT)'].mean()

print(f"Mean of CO(GT): {mean_co}")


# In[52]:


co_original = df['CO(GT)'].copy()

# Step 3: Apply linear interpolation
df['CO(GT)'] = df['CO(GT)'].interpolate(method='linear')

# Step 4: Prepare data for box plot
box_data = pd.DataFrame({
    'CO(GT)': pd.concat([co_original, df['CO(GT)']]),
    'Type': ['Before Interpolation'] * len(co_original) + ['After Interpolation'] * len(df['CO(GT)'])
})
# Drop NaN values (boxplot needs valid numbers)
box_data = box_data.dropna()

# Step 5: Plot box plot using Plotly
fig = px.box(box_data, x='Type', y='CO(GT)', title='CO(GT) Before vs After Interpolation')
fig.show()


# In[53]:


df = pd.read_csv("data/AmesHousing.csv")
df.head()


# In[54]:


#lot area , saleprice , garage area, gr liv area , year built


# In[56]:


df['Lot Area'].isnull().sum()


# In[57]:


df['Lot Area']


# In[58]:


df.info()


# In[65]:


df.isnull().sum()[:50]


# In[62]:


df.shape


# In[66]:


cols = ['Lot Area', 'SalePrice', 'Garage Area', 'Gr Liv Area', 'Year Built']
missing = df[cols].isnull().sum()
print("Missing values:\n", missing)


# In[70]:


#fill null values in Garage Area with median
df.fillna(df['Garage Area'].median(), inplace=True)


# In[81]:


#box plot for sale price color them based on year built
fig = px.box(df, y='SalePrice')
fig.show()


# In[75]:


fig = px.box(df, y='Garage Area')
fig.show()


# In[76]:


fig = px.box(df, y='Gr Liv Area')
fig.show()


# In[72]:


#Plot Gr Liv area vs sale price , repeat for garage area vs sale price , add trend line in both
fig = px.scatter(df, x='Gr Liv Area', y='SalePrice', trendline='ols')
fig.show()


# In[73]:


fig = px.scatter(df, x='Garage Area', y='SalePrice', trendline='ols')
fig.show()


# In[82]:


fig = px.scatter(df, x='Gr Liv Area', y='SalePrice', trendline='ols',color='Year Built')
fig.show()


# In[84]:


fig = px.scatter(df, x='Garage Area', y='SalePrice', trendline='ols',color = 'Year Built')
fig.show()


# In[86]:


#bmi vs medical charges
df = pd.read_csv("data/insurance.csv")
df.head()


# In[87]:


df.isnull().sum()


# In[88]:


fig = px.scatter(df, x='bmi', y='charges', trendline='ols',color = 'smoker')
fig.show()


# In[89]:


df.shape


# In[2]:


import plotly.express as px
import pandas as pd


# In[90]:


df.describe()


# In[93]:


fig = px.scatter(df, x='bmi', y='charges', trendline='ols',color = 'age')
fig.show()


# In[ ]:





# ## Day 2 — Analysis
# 
# *Original notebook: `Module03_Day2.ipynb`*
# 

# In[45]:


import pandas as pd
df = pd.read_csv("data/StudentsPerformance.csv")
df.head()


# In[ ]:


#Calculate the probability that a randomly selected student scored more than 80 in math.


# In[3]:


students_above_80_math = df[df['math score'] > 80].shape[0]

# Calculate the total number of students
total_students = df.shape[0]

# Calculate the probability
probability = students_above_80_math / total_students

print(f"The probability that a randomly selected student scored more than 80 in math is: {probability:.4f}")


# In[43]:


import pandas as pd
df = pd.read_csv("data/insurance.csv")
df.head()


# In[5]:


# Q-Q Plot Worked Example: Custom Data vs Theoretical Normal Quantiles

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------
# 1. Your dataset
# ----------------------------
data = np.array([55, 60, 62, 65, 68, 70, 75, 80, 85, 90])
n = len(data)

# ----------------------------
# 2. Calculate percentiles for each data point
# Formula: (i - 0.5) / n
# ----------------------------
percentiles = [(i - 0.5) / n for i in range(1, n+1)]

# ----------------------------
# 3. Calculate theoretical normal quantiles using inverse CDF (ppf)
# ----------------------------
expected_normal = [norm.ppf(p) for p in percentiles]

# ----------------------------
# 4. Plotting
# ----------------------------
plt.figure(figsize=(8,6))
plt.scatter(expected_normal, data, color='blue')
plt.title('Custom Q-Q Plot: Data vs Theoretical Normal Quantiles')
plt.xlabel('Theoretical Quantiles (Standard Normal)')
plt.ylabel('Data Values (Exam Scores)')

# Add a best fit line for reference
slope, intercept = np.polyfit(expected_normal, data, 1)
plt.plot(expected_normal, np.array(expected_normal)*slope + intercept, color='red', linestyle='--')

# Annotate each point with its percentile for teaching
for x, y, p in zip(expected_normal, data, percentiles):
    plt.text(x, y+0.5, f"{int(p*100)}%", fontsize=8)

plt.grid(True)
plt.tight_layout()
plt.show()


# In[10]:


#draw a qq plot for math score in the dataframe df
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Get the math scores and sort them
math_scores = np.sort(df['math score'])
n = len(math_scores)

# Calculate percentiles for each data point
percentiles = [(i - 0.5) / n for i in range(1, n+1)]

# Calculate theoretical normal quantiles using inverse CDF (ppf)
expected_normal = [norm.ppf(p) for p in percentiles]

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(expected_normal, math_scores, color='blue')
plt.title('Q-Q Plot: Math Score vs Theoretical Normal Quantiles')
plt.xlabel('Theoretical Quantiles (Standard Normal)')
plt.ylabel('Data Values (Math Scores)')

# Add a best fit line for reference
slope, intercept = np.polyfit(expected_normal, math_scores, 1)
plt.plot(expected_normal, np.array(expected_normal)*slope + intercept, color='red', linestyle='--')

# Annotate each point with its percentile for teaching
# Not practical for 1000 points, so I will remove this for clarity
# for x, y, p in zip(expected_normal, math_scores, percentiles):
#     plt.text(x, y+0.5, f"{int(p*100)}%", fontsize=8)

plt.grid(True)
plt.tight_layout()
plt.show()


# In[12]:


import plotly.express as px
px.histogram(df,'charges')


# In[15]:


px.box(df,'charges')


# In[17]:


# 2. Q-Q Plot using Matplotlib (scipy is used for quantiles)
import scipy.stats as stats
plt.figure(figsize=(6,6))
stats.probplot(df["charges"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Charges")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()


# In[18]:


df.shape


# In[19]:


df.head()


# In[20]:


df['region'].unique()


# In[29]:


import scipy.stats as stats
import matplotlib.pyplot as plt # Import matplotlib

plt.figure(figsize=(6,6))
smoker = df[df['smoker'] == 'yes']
non_smoker = df[df['smoker'] == 'no']
stats.probplot(smoker['charges'], dist="norm", plot=plt) # Select 'charges' column
stats.probplot(non_smoker['charges'], dist="norm", plot=plt) # Select 'charges' column
plt.title("Q-Q Plot of Charges")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()


# In[28]:


import scipy.stats as stats
import matplotlib.pyplot as plt # Import matplotlib

plt.figure(figsize=(6,6))
smoker = df[df['smoker'] == 'yes']
non_smoker = df[df['smoker'] == 'no']
stats.probplot(smoker['charges'], dist="norm", plot=plt) # Select 'charges' column
stats.probplot(non_smoker['charges'], dist="norm", plot=plt) # Select 'charges' column
plt.title("Q-Q Plot of Charges")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()


# In[ ]:





# In[31]:


import scipy.stats as stats
import matplotlib.pyplot as plt

# Prepare data
smoker = df[df['smoker'] == 'yes']['charges']
non_smoker = df[df['smoker'] == 'no']['charges']

# Generate Q-Q data
smoker_qq = stats.probplot(smoker, dist="norm")
non_smoker_qq = stats.probplot(non_smoker, dist="norm")

# Create plot
plt.figure(figsize=(6,6))

# Plot smokers' Q-Q plot and line
plt.plot(smoker_qq[0][0], smoker_qq[0][1], 'o', label='Smokers', color='red')
smoker_line = smoker_qq[1][1] + smoker_qq[1][0] * smoker_qq[0][0]
plt.plot(smoker_qq[0][0], smoker_line, 'r-', alpha=0.5)

# Plot non-smokers' Q-Q plot and line
plt.plot(non_smoker_qq[0][0], non_smoker_qq[0][1], 'x', label='Non-Smokers', color='blue')
non_smoker_line = non_smoker_qq[1][1] + non_smoker_qq[1][0] * non_smoker_qq[0][0]
plt.plot(non_smoker_qq[0][0], non_smoker_line, 'b-', alpha=0.5)

# Plot formatting
plt.title("Q-Q Plot of Charges: Smokers vs Non-Smokers")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.legend()
plt.show()


# In[32]:


df = pd.read_csv("data/Mall_Customers.csv")
df.head()


# In[35]:


px.scatter(df,'Annual Income (k$)','Spending Score (1-100)',color='Gender',trendline='ols')


# Hypothsis Testing

# In[38]:


# 1 tailed t test
from scipy import stats

# Sample delivery times (in minutes)
delivery_times = [32, 35, 30, 31, 36, 33, 29, 34, 37, 28]

# Population mean to test against
mu_0 = 30

# One-sample t-test (one-tailed test: greater than 30)
t_stat, p_value_two_tailed = stats.ttest_1samp(delivery_times, mu_0)

# For one-tailed test (greater than), divide p-value by 2
p_value_one_tailed = p_value_two_tailed / 2

# Output results
print("Sample Mean:", round(sum(delivery_times) / len(delivery_times), 2))
print("t-statistic:", round(t_stat, 3))
print("p-value (one-tailed):", round(p_value_one_tailed, 4))

# Conclusion
alpha = 0.05
if p_value_one_tailed < alpha and t_stat > 0:
    print("Conclusion: Reject H0 — average delivery time is greater than 30 minutes.")
else:
    print("Conclusion: Fail to reject H0 — not enough evidence that average is greater.")


# In[39]:


#Independent 2 tail test
from scipy import stats

# Scores from two independent groups
group_a = [75, 78, 74, 72, 80, 77, 73, 76, 79, 74]  # Traditional
group_b = [82, 85, 84, 81, 86, 83, 80, 87, 85, 84]  # Interactive

# Perform two-sample t-test (equal variances assumed)
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True)

# Output results
print("Group A Mean:", round(sum(group_a)/len(group_a), 2))
print("Group B Mean:", round(sum(group_b)/len(group_b), 2))
print("t-statistic:", round(t_stat, 3))
print("p-value:", round(p_value, 4))

# Interpret result
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject H0 — There is a significant difference between the teaching methods.")
else:
    print("Conclusion: Fail to reject H0 — No significant difference detected.")


# In[40]:


#Anova

from scipy import stats

# Step 1: Create test scores for each group
group_a = [70, 72, 68, 75, 74]  # Traditional
group_b = [80, 82, 85, 79, 81]  # Online
group_c = [90, 88, 92, 91, 89]  # Workshop

# Step 2: Perform one-way ANOVA test
f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)

# Step 3: Print the result
print("F-statistic:", round(f_stat, 2))
print("p-value:", round(p_value, 4))

# Step 4: Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject H0 — At least one group is different.")
else:
    print("Conclusion: Fail to reject H0 — No significant difference between groups.")


# In[42]:


#Chi Square
import numpy as np
from scipy.stats import chi2_contingency

# Observed frequency table (2x2)
# Rows = Gender (Male, Female)
# Columns = Purchase (Yes, No)

observed = np.array([
    [30, 20],  # Male: 30 Yes, 20 No
    [10, 40]   # Female: 10 Yes, 40 No
])


# In[ ]:


# Shapiro-Wilk test on df


# In[44]:


from scipy.stats import shapiro

# Perform the Shapiro-Wilk test on the 'charges' column
shapiro_test_statistic, shapiro_p_value = shapiro(df['charges'])

# Output the results
print(f"Shapiro-Wilk Test Statistic: {shapiro_test_statistic:.4f}")
print(f"Shapiro-Wilk P-value: {shapiro_p_value:.4f}")

# Interpret the result
alpha = 0.05
if shapiro_p_value < alpha:
    print("Conclusion: Reject H0 - The 'charges' data does not appear to be normally distributed.")
else:
    print("Conclusion: Fail to reject H0 - The 'charges' data appears to be normally distributed.")


#  3. One-tail t-test Question
# 
# Dataset: [Students Performance in Exams (same as above)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
# 
# Problem:
# Test the hypothesis that mean math score is greater than 65 (use one-sample, one-tail t-test).
# 
# Steps for Students:
# 
#  Null hypothesis: mean ≤ 65
#  Alternative hypothesis: mean > 65
#  Conduct one-sample t-test
#  Report t-statistic and p-value; conclude
# 
# 

# In[46]:


df.head()


# In[48]:


# Null hypothesis: mean ≤ 65 Alternative hypothesis: mean > 65 Conduct one-sample t-test Report t-statistic and p-value; conclude
from scipy import stats
t_stat, p_value_two_tailed = stats.ttest_1samp(df['math score'], 65)
p_value_one_tailed = p_value_two_tailed / 2
print("Sample Mean:", round(sum(df['math score']) / len(df['math score']), 2))
print("t-statistic:", round(t_stat, 3))
print("p-value (one-tailed):", round(p_value_one_tailed, 4))
alpha = 0.05
if p_value_one_tailed < alpha and t_stat > 0:
    print("Conclusion: Reject H0 — average math score is greater than 65.")
else:
    print("Conclusion: Fail to reject H0 — not enough evidence that average is greater.")


#  4. Two-tail t-test Question
# 
# Dataset: [Students Performance in Exams (same as above)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
# 
# Problem:
# Test if average math score differs between male and female students (two-sample, two-tail t-test).
# 
# Steps for Students:
# 
#  Null hypothesis: means are equal
#  Alternative hypothesis: means are different
#  Perform independent two-sample t-test
#  Report t-statistic and p-value; conclude
# 
# 

# In[49]:


#Null hypothesis: means are equal Alternative hypothesis: means are different Perform independent two-sample t-test Report t-statistic and p-value; conclude
t_stat, p_value = stats.ttest_ind(df[df['gender'] == 'male']['math score'], df[df['gender'] == 'female']['math score'], equal_var=True)
print("t-statistic:", round(t_stat, 3))
print("p-value:", round(p_value, 4))
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject H0 — There is a significant difference between the average math scores of male and female students.")
else:
    print("Conclusion: Fail to reject H0 — No significant difference detected.")


# In[50]:


df = pd.read_csv('data/ds_salaries.csv')
df.head()


# In[ ]:


#does size of the company affects the salary


# **Hypothesis Test: Salary vs. Company Size**
# 
# *   **Null Hypothesis (H0):** The mean salary is the same across different company sizes.
# *   **Alternative Hypothesis (H1):** The mean salary is different for at least one company size.

# In[51]:


from scipy import stats

# Separate salaries by company size
salary_s = df[df['company_size'] == 'S']['salary_in_usd']
salary_m = df[df['company_size'] == 'M']['salary_in_usd']
salary_l = df[df['company_size'] == 'L']['salary_in_usd']

# Perform one-way ANOVA test
f_stat, p_value = stats.f_oneway(salary_s, salary_m, salary_l)

# Output results
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject H0 - There is a significant difference in average salary for at least one company size.")
else:
    print("Conclusion: Fail to reject H0 - No significant difference in average salary detected across company sizes.")


# In[52]:


df1 = pd.read_csv('data/tmdb_5000_movies.csv')
df1.head()


# In[ ]:


#


# In[55]:


import json

# Function to extract genre names
def get_genres(genre_list):
    if isinstance(genre_list, str):
        genres = json.loads(genre_list)
        return [g['name'] for g in genres]
    return []

# Apply the function to create a list of genre names for each movie
df1['genre_names'] = df1['genres'].apply(get_genres)

# Create a new dataframe with one row per genre for each movie
df_genres = df1.explode('genre_names')

# Group by genre and calculate the mean revenue
genre_revenue = df_genres.groupby('genre_names')['revenue'].mean().sort_values(ascending=False)

# Convert revenue to millions
genre_revenue_in_millions = genre_revenue / 1_000_000

# Display the result in millions
print("Average Revenue by Genre (in millions):")
print(genre_revenue_in_millions)


# In[ ]:





# In[56]:


import matplotlib.pyplot as plt

# Create a bar chart
plt.figure(figsize=(12, 8))
genre_revenue_in_millions.plot(kind='bar')

# Add labels and title
plt.xlabel("Genre")
plt.ylabel("Average Revenue (in millions)")
plt.title("Average Movie Revenue by Genre")
plt.xticks(rotation=90) # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()


# **Hypothesis Test: Revenue vs. Genre**
# 
# *   **Null Hypothesis (H0):** The mean revenue is the same across all movie genres.
# *   **Alternative Hypothesis (H1):** The mean revenue is different for at least one movie genre.

# In[57]:


from scipy import stats

# Prepare data for ANOVA - create a list of revenue arrays for each genre
# We'll use the df_genres dataframe created earlier which has one row per genre per movie
genre_groups = [df_genres[df_genres['genre_names'] == genre]['revenue'].dropna() for genre in df_genres['genre_names'].unique()]

# Remove any empty arrays that might result from genres with no revenue data
genre_groups = [group for group in genre_groups if not group.empty]

# Perform one-way ANOVA test
f_stat, p_value = stats.f_oneway(*genre_groups)

# Output results
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject H0 - There is a significant difference in average revenue for at least one genre.")
else:
    print("Conclusion: Fail to reject H0 - No significant difference in average revenue detected across genres.")


# In[60]:


import plotly.express as px

# Use the df_genres dataframe which has been exploded by genre
fig = px.box(df_genres, x='genre_names', y='revenue', title='Box Plot of Movie Revenue by Genre')
fig.show()


# In[ ]:





# In[ ]:





# In[61]:


# Calculate the number of genres for each movie
df1['num_genres'] = df1['genre_names'].apply(len)

# Display the distribution of the number of genres
print("Distribution of the number of genres per movie:")
print(df1['num_genres'].value_counts().sort_index())

# Display the average number of genres per movie
average_genres = df1['num_genres'].mean()
print(f"\nAverage number of genres per movie: {average_genres:.2f}")


# In[77]:


# Get the value counts of the 'genre_names' list
genre_combination_counts = df1['genre_names'].value_counts()

# Display the top 10 genre combinations
print("Top 10 Genre Combinations:")
print(genre_combination_counts)


# In[ ]:





# In[63]:


import plotly.express as px

# Get the top 10 genre combinations (assuming genre_combination_counts is already calculated)
top_10_genre_combinations = genre_combination_counts.head(10).index.tolist()

# Filter the dataframe to include only movies with one of the top 10 genre combinations
df_top_genres = df1[df1['genre_names'].apply(lambda x: x in top_10_genre_combinations)]

# Create a box plot for the filtered data
fig = px.box(df_top_genres, x='genre_names', y='revenue', title='Box Plot of Movie Revenue for Top 10 Genre Combinations')
fig.show()


# In[ ]:





# In[68]:


# Create a new column with genre names joined by a space
df1['genre_string'] = df1['genre_names'].apply(lambda x: ' '.join(x))

# Display the head of the dataframe to show the new column
display(df1.head())


# In[69]:


import plotly.express as px

# Get the top 10 most frequent genre strings
top_10_genre_strings = df1['genre_string'].value_counts().head(10).index.tolist()

# Filter the dataframe to include only movies with one of the top 10 genre strings
df_top_genre_strings = df1[df1['genre_string'].isin(top_10_genre_strings)]

# Create a box plot for the filtered data
fig = px.box(df_top_genre_strings, x='genre_string', y='revenue', title='Box Plot of Movie Revenue for Top 10 Genre String Combinations')
fig.show()


# In[76]:


df1['genre_string'].nu


# In[ ]:




