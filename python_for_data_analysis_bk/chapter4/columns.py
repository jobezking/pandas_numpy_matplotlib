import pandas as pd

df = pd.DataFrame({
    'Name':['Alice','Bob','Charlie','David'],
    'Age':[25,30,35,40],
    'City':['New York', 'Los Angeles', 'Chicago', 'Houston']
})

#Select column
ages = df['Age']
print(ages)

print(df[['Name','City']])

first_three_rows = df.iloc[0:3]
print(first_three_rows)

alice_row = df.loc['Alice'] 
print(alice_row)

alice_age = df.loc['Alice', 'Age'] 
print(alice_age)

subset = df.loc[['Alice', 'David'], ['Age', 'City']] 
print(subset)

older_than_30 = df[df['Age'] > 30] 
print(older_than_30)

filtered = df[(df['Age'] > 25) & (df['City'] == 'Chicago')] 
print(filtered)

cities = ['New York', 'Houston'] 
selected_cities = df[df['City'].isin(cities)] 
print(selected_cities)

df_reset = df.reset_index() 
print(df_reset)

data = { 
        'Customer': ['Anna', 'Brian', 'Catherine', 'Daniel', 'Eva'], 
        'Age': [28, 34, 22, 45, 31],
        'City': ['New York', 'Chicago', 'Los Angeles', 'Chicago', 'New York'], 
        'Sales': [250, 400, 150, 300, 500] 
}

df = pd.DataFrame(data) # Filter customers older than 30 and in either New York or Chicago 
filtered_customers = df[(df['Age'] > 30) & (df['City'].isin(['New York', 'Chicago']))] 
print(filtered_customers)

data = {     
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],     
    'Age': [25, 30, 35, 40, 22],     
    'Sales': [200, 450, 300, 500, 150] 
}   
df = pd.DataFrame(data)   # Sort by Age ascending (default) 
sorted_by_age = df.sort_values('Age') 
print(sorted_by_age)

sorted_by_sales_desc = df.sort_values('Sales', ascending=False) 
print(sorted_by_sales_desc)

sorted_multiple = df.sort_values(['Age', 'Sales'], ascending=[True, False]) 
print(sorted_multiple)

df_indexed = df.set_index('Name') 
sorted_index = df_indexed.sort_index() 
print(sorted_index)

df['Sales_rank'] = df['Sales'].rank() 
print(df)

df['Sales_rank_min'] = df['Sales'].rank(method='min') 
df['Sales_rank_first'] = df['Sales'].rank(method='first') 
print(df)

df['Sales_rank_desc'] = df['Sales'].rank(ascending=False)
print(df)

# Sort by Sales descending, then Age ascending 
sorted_sales = df.sort_values(['Sales', 'Age'], ascending=[False, True])   

# Add ranks for sales

df['Sales_rank'] = df['Sales'].rank(ascending=False, method='min')   
print(sorted_sales) 
print(df)
