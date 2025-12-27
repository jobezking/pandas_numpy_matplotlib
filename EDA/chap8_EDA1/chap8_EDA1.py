import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

#Load Titanic dataset directly from seaborn
df = sns.load_dataset('titanic')
df.to_csv('titanic.csv')

titanic = pd.read_csv('titanic.csv')
titanic.head()

titanic.info()
print(titanic.shape)
titanic.describe()
titanic.isna().sum()

titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['embark_town'] = titanic['embark_town'].fillna(titanic['embark_town'].mode()[0])
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
titanic['deck'] = titanic['deck'].fillna(titanic['deck'].mode()[0])

print(titanic.info())
print(titanic.describe())

num_features = titanic.select_dtypes(include=['int64','float64']).columns
cat_features = titanic.select_dtypes(include=['object']).columns

sns.set(style="whitegrid")
sns.countplot(x='survived',data=titanic)
plt.title('Overall Survival Distribution')
plt.show()
