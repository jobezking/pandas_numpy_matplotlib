import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic') #Load example dataset.
df.head() #Display the first few rows of the dataset.
df.info() #Get a summary of the dataset including data types and non-null counts. Pick features and target variable.
df.describe() #Get statistical summary of numerical features. 
#probability distribtion of age variable
sns.histplot(df['age'],kde=True)  #kde is kernel density estimate curve to show probability distribution
plt.title('Age Distribution of Passengers')
plt.show()
#distribution plot of sex variable
sns.countplot(data=df, x='sex')
plt.title('Sex Distribution of Passengers')
plt.show()
#plot survived vs sex
sns.barplot(x='sex', y='survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()
#plot survived vs class
sns.barplot(x='pclass',y='survived',data=df)
plt.title('Survival Rate by Class')
plt.show()
#plot survived vs sex and class
sns.catplot(x='pclass',y='survived',hue='sex',data=df, kind='bar')
plt.title('Titanic Survival Rate by Class and Sex')
plt.show()
#Show missing data by variable
sns.heatmap(df.isnull(),cbar=False)
plt.title('Missing Values in Titanic Dataset')
plt.show()

df.dtypes #Check data types of each column.
df['column'] = df['column'].astype('float') #Convert a column to a different data type.
df.isnull().sum() #Check for missing values in each column.
df.dropna(inplace=True) #Remove rows with missing values.
df['age'] = df['age'].fillna(df['age'].median()) #Fill missing values in 'age' column with median age.
df.duplicated().sum() #Check for duplicate rows in the dataset.
df.drop_duplicates(inplace=True) #Remove duplicate rows from the dataset.

#Visualization statistics
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1 # interquartile range
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['fare'] < lower) | (df['fare'] > upper)]
#visualize outliers
sns.boxplot(x=df['fare'])
plt.title('Boxplot of Fare with Outliers')
plt.show()
#
sns.scatterplot(data=df, x=df.index, y="fare", color="gray", alpha=0.6, label="Normal") 
sns.scatterplot(data=outliers, x=outliers.index, y="fare", color="red", s=80, label="Outliers") 
plt.title("Outliers Highlighted in Red") 
plt.legend() 
plt.show()
#
sns.histplot(df["fare"], bins=30, kde=True)
plt.axvline(lower, color="red", linestyle="--", label="Lower bound") 
plt.axvline(upper, color="red", linestyle="--", label="Upper bound") 
plt.title("Fare Distribution with Outlier Thresholds") 
plt.legend() 
plt.show()
#Remove outliers
df = df[(df['fare'] >= lower) & (df['fare'] <= upper)]