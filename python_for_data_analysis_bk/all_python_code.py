import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#note: chapter 4 is very important for pandas fundamentals
#note: chapter 5 is very important for data cleaning
data = {
    'Name':['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age':[25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}
df = pd.DataFrame(data)
nums = np.array([10,20,30,40,50])
print(np.mean(nums))

print(df.info())
print(df.head())
df.to_csv('cleaned_data.csv',index=False)

excel_data = pd.ExcelFile('email_logistic_models.xlsx')
print(excel_data.sheet_names)
df_excel = excel_data.parse('Stats 2')
print(df_excel.head())
df_excel.to_excel('summary_report.xlsx',sheet_name='Summary',index=False)

df2 = pd.read_excel('email_logistic_models.xlsx',sheet_name='Model 2')
print(df2.head())

#open json file
with open('a_sample.json','r') as file:
    data = json.load(file)
print(data)
#save to json file
with open('output.json','w') as file:
    json.dump(data,file,indent=4)
#If json data is flat and tabular, pandas can read it directly
df_json = pd.read_json('output.json')
df_json.to_json('data.json',orient='records')

#Matplotlib
df_data = sns.load_dataset("sales_data.csv")
df_data = pd.read_csv("sales_data.csv")
print(df_data.head())
print(f"Dataset contains {df_data.shape[0]} rows and {df_data.shape[1]} columns.\n")
df_data.info()
print("\n")
df_data.describe()
print(df_data['CategoryColumn'].describe())
print(df_data['Region'].value_counts())
print(df_data.isnull().sum())
df_data['quantity'].hist(bins=30) 
plt.title('Sales Distribution')
plt.xlabel('Sales') 
plt.ylabel('Frequency') 
plt.show()

df_cat = pd.DataFrame({'Color': ['red', 'blue', 'green', 'blue', 'red']})

#One-Hot Encoding
one_hot = pd.get_dummies(df_cat['Color'],prefix='Color')

#Label Encoding
label_encoder = LabelEncoder()
df_cat['Color_label'] = label_encoder.fit_transform(df_cat['Color'])

print("One-Hot Encoding:\n", one_hot) 
print("\nLabel Encoding:\n", df_cat)

modeldata = sns.load_dataset("sales_data_old.csv") #load data
modeldata_encoded = pd.get_dummies(modeldata, columns=['category'], drop_first=True) #encode category one hot
modeldata = pd.concat([modeldata, modeldata_encoded], axis=1) # merge unencoded and encoded dataframe
modeldata           #print dataframe
y = modeldata['price']  #y axis, y = f(x), or labels
X = modeldata[['quantity', 'revenue', 'category_Electronics', 'category_Clothing', 'category_Shoes']] #features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split into training and testing sets
model = RandomForestClassifier(n_estimators=100, random_state=42) # Step 3: initialize model
# model = LinearRegression()
model.fit(X_train, y_train) # train the model
y_pred = model.predict(X_test) # make predictions

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared value
r_squared = r2_score(y_test, y_pred)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Test values (provided for you)
print(f"Your Mean Squared Error (MSE): {mse}")

print(f"Your R-squared: {r_squared}")

print(f"Your Mean Absolute Error (MAE): {mae}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Y Values")
plt.ylabel("Predicted Y Values")
plt.title("Actual vs. Predicted Y Values")
plt.grid(True)
plt.show()

def summarize_data(arr):
    print(f"Mean: {np.mean(arr)}") 
    print(f"Median: {np.median(arr)}")  
    print(f"Variance: {np.var(arr)}")  
    print(f"Standard Deviation: {np.std(arr)}") 
    print(f"Min: {np.min(arr)}")  
    print(f"Max: {np.max(arr)}")  
    print(f"Range: {np.max(arr) - np.min(arr)}")
    print(f"25th Percentile (Q1): {np.percentile(arr, 25)}")  
    print(f"75th Percentile (Q3): {np.percentile(arr, 75)}")

arra = np.array([10, 20, 30, 40, 50])
summarize_data(arra)