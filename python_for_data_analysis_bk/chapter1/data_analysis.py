import pandas as pd

# Create a small sample dataset
data = {
    'Name':['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age':[25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

# Display basic information about the Dataframe
print("Summary of the data:")
print(df.info())

print("\nFirst few rows:")
print(df.head())