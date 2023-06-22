import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

rf = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier(random_state=1)

df = pd.read_csv("D:/CSV Files/HousingData.csv")
print(df.keys())

le = LabelEncoder()
le.fit(df['MEDV'])
df['MEDV'] = le.transform(df['MEDV'])
print(df)

df = df.drop('CHAS', axis=1)
df = df.drop('ZN', axis=1)
df = df.drop('RAD', axis=1)
df = df.drop('PTRATIO', axis=1)
df = df.drop('TAX', axis=1)
X = df.drop('MEDV', axis=1)
Y = df['MEDV']

print(X.isnull().sum())

X['CRIM'].fillna((X['CRIM'].mean()), inplace=True)
X['INDUS'].fillna((X['INDUS'].mean()), inplace=True)
X['AGE'].fillna((X['AGE'].mean()), inplace=True)
X['LSTAT'].fillna((X['LSTAT'].mean()), inplace=True)

print(X.isnull().sum())

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(13).plot(kind='barh')
plt.show()

from collections import Counter
print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X,Y)
print(Counter(Y))

import seaborn as sns
sns.boxplot(df['LSTAT'])
plt.show()
out = ['DIS','LSTAT','RM','CRIM']
for i in out:
    print(X[i])
    Q1 = X[i].quantile(0.25)
    Q3 = X[i].quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    print(upper)
    print(lower)
    out1 = X[X[i] < lower].values
    out2 = X[X[i] > upper].values
    X[i].replace(out1, lower, inplace=True)
    X[i].replace(out2, upper, inplace=True)
    sns.boxplot(X[i])
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.3)

func = [rf, dtc]
for item in func:
    item.fit(X_train, y_train)
    y_pred = item.predict(X_test)
    print(item)
    print(mean_squared_error(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
