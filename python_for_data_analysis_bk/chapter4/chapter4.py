import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sales_data.csv')
print(df.head())
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
df.info()
print(df.describe())
print(df['CategoryColumn'].describe())
print(df['Region'].value_counts())
print(df.isnull().sum())
df['Sales'].hist(bins=30) 
plt.title('Sales Distribution')
plt.xlabel('Sales') 
plt.ylabel('Frequency') 
plt.show()

df = pd.read_excel('financial_report.xlsx', sheet_name='Q1')
print(df.head())

df = pd.read_json('user_data.json')
print(df.head())

df.to_csv('cleaned_data.csv',index=False)
df.to_excel('summary_report.xlsx',sheet_name='Summary',index=False)
df.to_json('data.json',orient='records')

