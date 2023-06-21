
import pandas as pd
df = pd.read_csv("D:/CSV Files/titanic_data.csv")

# Identifying the Outliers by plotting

from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['Survived'])
plt.show()

# Dealing with the Outliers using InterQuantile Range

print(df['Survived'])
Q1 = df['Survived'].quantile(0.25)
Q3 = df['Survived'].quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper =Q3 + 1.5*IQR
lower =Q1 - 1.5*IQR

print(upper)
print(lower)

out1 = df[df['Survived']< lower].values
out2 = df[df['Survived']>upper].values

df['Survived'].replace(out1,lower,inplace = True)
df['Survived'].replace(out2,upper,inplace = True)

print(df['Survived'])

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr = LogisticRegression()
pca = PCA(n_components=2)

X= df.drop('Survived',axis=1)
X= X.drop('Embarked',axis=1)
Y=df['Embarked']

pca.fit(X)
X = pca.transform(X)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(accuracy_score(y_test,y_pred))