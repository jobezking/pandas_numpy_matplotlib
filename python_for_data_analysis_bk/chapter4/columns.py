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