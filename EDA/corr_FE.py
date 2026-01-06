import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
#FE
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

#CORR
titanic = pd.read_csv('titanic_cleaned2.csv')
corr = titanic.corr(numeric_only = True)
print(corr['survived'])

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

#USe numeric only for now
titanicc = titanic.select_dtypes(include=['int64', 'float64'])  #get numeric features only for now
X = titanicc.drop('survived', axis=1) 
y = titanicc['survived']
#X = titanic.drop('survived', axis=1)
#y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
print(coefficients.head(10))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train) rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))