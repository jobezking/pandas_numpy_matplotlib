from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = fetch_california_housing(as_frame=True)
df = data.frame

df['MedInc'] = df['MedInc'].fillna(df['MedInc'].median())
df['HouseAge'] = df['HouseAge'].fillna(df['HouseAge'].median())
#or
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#or
iter_imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(iter_imputer.fit_transform(df), columns=df.columns)
#
z_scores = np.abs(stats.zscore(df['MedInc'],nan_policy='omit'))
outliers = df[z_scores > 3]
#or
Q1 = df['MedInc'].quantile(0.25)
Q3 = df['MedInc'].quantile(0.75)
IQR = Q3 - Q1
outliers2 = df[(df['MedInc'] < (Q1 - 1.5 * IQR)) | (df['MedInc'] > (Q3 + 1.5 * IQR))]
#
upper_limit = df['MedInc'].quantile(0.99)
lower_limit = df['MedInc'].quantile(0.01)
df['MedInc'] = np.where(df['MedInc'] > upper_limit, upper_limit, 
                        np.where(df['MedInc'] < lower_limit, lower_limit, df['MedInc']))

#Example
# Step 1: Create missing values 
df.loc[df.sample(frac=0.05).index, 'MedInc'] = np.nan
# Step 2: Impute missing values using KNN
imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# Step 3: Detect outliers with Z-score
z_scores = np.abs(stats.zscore(df_imputed['MedInc']))
outliers = df_imputed[z_scores > 3]
print(f"Detected {len(outliers)} outliers.")
# Step 4: Cap extreme values
upper = df_imputed['MedInc'].quantile(0.99)
lower = df_imputed['MedInc'].quantile(0.01)
df_imputed['MedInc'] = np.clip(df_imputed['MedInc'], lower, upper)
print(df_imputed.isnull().sum())
print(df_imputed.describe())

df_imputed.head()