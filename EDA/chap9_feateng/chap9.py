import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

titanic = pd.read_csv('titanic_cleaned.csv')

#Create combined features
titanic['FamilySize'] = titanic['sibsp'] + titanic['parch'] + 1

titanic['IsAlone'] = 1  #sets every column to traveling alone because values will be either 1 or 0
titanic.loc[titanic['FamilySize'] > 1, 'IsAlone'] = 0 # if family size is greater than 1, not alone. set field to 0

#use log for skewed distributions
titanic['fare'] = np.log1p(titanic['fare'])
titanic['Age'] = np.log1p(titanic['age'])

titanic = pd.get_dummies(titanic, columns=['sex', 'embarked', 'pclass'], drop_first=True) #categorical

scaler = StandardScaler()
num_features = ['age', 'fare', 'FamilySize']

titanic[num_features] = scaler.fit_transform(titanic[num_features])

titanic.head()

titanic.info()

titanic.to_csv('titanic_cleaned2.csv')
