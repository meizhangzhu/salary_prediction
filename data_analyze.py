import code
import json

import numpy as np
import pandas as pd
import seaborn as sns
# import xgboost as xgb

import matplotlib.pyplot as plt
from scipy import stats as scs
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold

features = ["companyId_n", "jobType_n", "degree_n", "major_n", "industry_n"]

df_x = pd.read_csv("train_features.csv")
df_y = pd.read_csv("train_salaries.csv")
df = df_x.join(df_y, lsuffix = '_x', rsuffix = '_y')
np_df = np.array(df_x)

# code.interact(local = locals())

jobId, companyId, jobType, degree, major, industry, yearsExperience, milesFromMetropolis = np_df.T

# correlations.
corr_year_salary = df['yearsExperience'].corr(df['salary'])
corr_mile_salary = df['milesFromMetropolis'].corr(df['salary'])

print("yearsExperience and salary: {}, positive moderate linear relative.".format(corr_year_salary))
print("milesFromMetropolis and salary: {}, negative weak linear relative.".format(corr_mile_salary))


fig = plt.figure(figsize=(12,12))
ax = sns.regplot(x='salary',y='milesFromMetropolis',data=df,scatter_kws={"s": 5})
plt.savefig('mfm_salary_regplot.png')

fig = plt.figure(figsize=(12,12))
ax = sns.regplot(x='salary',y='yearsExperience',data=df,scatter_kws={"s": 5})
plt.savefig('mfm_salary_regyears.png')

fig = plt.figure(figsize=(20,20))
ax = sns.lmplot(x='salary',y='yearsExperience',data=df,hue='degree', scatter_kws={'s':5.0},legend=False,fit_reg=False,)
plt.legend(loc=2,prop={'size':6})
plt.savefig('ye_salary_degree_lmplot.png')

fig = plt.figure(figsize=(20,20))
ax = sns.lmplot(x='salary',y='yearsExperience',data=df,hue='major',\
                scatter_kws={'s':5.0},legend=False,fit_reg=False,)
plt.legend(loc=2,prop={'size':6})
plt.savefig('ye_salary_major_lmplot.png')

fig = plt.figure(figsize=(20,20))
ax = sns.lmplot(x='salary',y='yearsExperience',data=df,hue='jobType',\
                scatter_kws={'s':5.0},legend=False,fit_reg=False,)
plt.legend(loc=2,prop={'size':6})
plt.savefig('ye_salary_jobType_lmplot.png')

fig = plt.figure(figsize=(20,20))
ax = sns.lmplot(x='salary',y='yearsExperience',data=df,hue='industry',\
                scatter_kws={'s':5.0},legend=False,fit_reg=False,)
plt.legend(loc=2,prop={'size':6})
plt.savefig('ye_salary_industry_lmplot.png')

fig = plt.figure(figsize=(20,20))
ax = sns.lmplot(x='salary',y='yearsExperience',data=df,hue='companyId',\
                scatter_kws={'s':5.0},legend=False,fit_reg=False,)
plt.legend(loc=2,prop={'size':6})
plt.savefig('ye_salary_companyId_lmplot.png')

fig = plt.figure(figsize=(12,12))
ax = sns.boxplot(x='salary',y='jobType',data=df,orient='h')
plt.savefig('jobtype_salary_boxplot.png')

fig = plt.figure(figsize=(12,12))
ax = sns.boxplot(x='salary',y='companyId',data=df,orient='h')
plt.savefig('companyId_salary_boxplot.png')

fig = plt.figure(figsize=(12,12))
ax = sns.boxplot(x='salary',y='degree',data=df,orient='h')
plt.savefig('degree_salary_boxplot.png')

fig = plt.figure(figsize=(12,12))
ax = sns.boxplot(x='salary',y='major',data=df,orient='h')
plt.savefig('major_salary_boxplot.png')

fig = plt.figure(figsize=(12,12))
ax = sns.boxplot(x='salary',y='industry',data=df,orient='h')
plt.savefig('industry_salary_boxplot.png')

fig = plt.figure(figsize=(12,12))
ax = sns.boxplot(x='salary',y='yearsExperience',data=df,orient='h')
plt.savefig('yearsExperience_salary_boxplot.png')

fig = plt.figure(figsize=(12,12))
ax = sns.boxplot(x='salary',y='milesFromMetropolis',data=df,orient='h')
plt.savefig('milesFromMetropolis_salary_boxplot.png')

fig = plt.figure(figsize = (12, 12))
ax = sns.countplot(y="companyId", data=df)
plt.savefig('companyId_count.png')

fig = plt.figure(figsize = (12, 12))
ax = sns.countplot(y="jobType", data=df)
plt.savefig('jobType_count.png')

fig = plt.figure(figsize = (12, 12))
ax = sns.countplot(y="degree", data=df)
plt.savefig('degree_count.png')

fig = plt.figure(figsize = (12, 12))
ax = sns.countplot(y="major", data=df)
plt.savefig('major_count.png')

fig = plt.figure(figsize = (12, 12))
ax = sns.countplot(y="industry", data=df)
plt.savefig('industry_count.png')

fig = plt.figure(figsize=(12,12))
ax = sns.distplot(df['salary'])
plt.savefig('salary_hist.png')

fig = plt.figure(figsize=(12,12))
ax = sns.distplot(df['yearsExperience'])
plt.savefig('yearsExperience_hist.png')

fig = plt.figure(figsize=(12,12))
ax = sns.distplot(df['milesFromMetropolis'])
plt.savefig('milesFromMetropolis_hist.png')


code.interact(local = locals())
