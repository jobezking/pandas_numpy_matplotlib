import pandas as pd

# Load the CSV file into a DataFrame
data = pd.read_csv("sales_data.csv")

# Preview the first few rows
print(data.head())

#Save dataframe to new CSV file
data.to_csv('processed_sales_data.csv')

#Load entire Excel file
excel_data = pd.ExcelFile('email_logistic_models.xlsx')

#List all sheet names
print(excel_data.sheet_names)
# returns ['email', 'Stats 1', 'Stats 2', 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model.5.CV.table', 'Model Summaries']

#Load specific sheet into a DataFrame
df = excel_data.parse('Stats 2')
print(df.head())

#Or you can directly load it:
df2 = pd.read_excel('email_logistic_models.xlsx',sheet_name='Model 2')
print(df2.head())

#Save dataframe back to Excel:
df2.to_excel('email_logistic_model_2.xlsx')