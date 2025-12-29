# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:37:12 2025

@author: LAB
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
data={
      'Age':[25,30,28,35,40,29,60],
      'Salary':[50000,60000,np.nan,80000,90000,70000,150000],
      'Department':['IT','Marketing','IT',np.nan,'HR','Marketing','IT'],
      'Score':[85,78,92,90,70,120,80]
      }
df=pd.DataFrame(data)
print("Original Dataset: \n",df)
selected_attributes=df[['Age','Salary']]
print("\n Selected Attributes:\n ",selected_attributes)
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])
print("\n Dataset After Hnadling Missing values: \n",df)
age_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df['Age_binned'] = age_discretizer.fit_transform(df[['Age']])
print("\n Dataset After Discretization (age Binned) :\n ",df)
Q1=df['Score'].quantile(0.25)
Q3=df['Score'].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
df_outliers_removed=df[(df['Score']>=lower_bound)&(df['Score']<=upper_bound)]
print("DataSet After Removing Outliers(Score): \nb ",df_outliers_removed)

