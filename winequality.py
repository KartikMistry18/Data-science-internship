
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import  GradientBoostingClassifier

rf = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier(random_state=1)
df = pd.read_csv('D:/CSV Files/winequalityN.csv')
print(df.keys())

print(df.isnull().sum())

df['fixed acidity'].fillna((df['fixed acidity'].mean()), inplace=True)
df['volatile acidity'].fillna((df['volatile acidity'].mean()), inplace=True)
df['citric acid'].fillna((df['citric acid'].mean()), inplace=True)
df['residual sugar'].fillna((df['residual sugar'].mean()), inplace=True)
df['chlorides'].fillna((df['chlorides'].mean()), inplace=True)
df['pH'].fillna((df['pH'].mean()), inplace=True)
df['sulphates'].fillna((df['sulphates'].mean()), inplace=True)

print(df.isnull().sum())

le = LabelEncoder()
le.fit(df['type'])
df['type'] = le.transform(df['type'])

X = df.drop('quality', axis=1)
Y = df['quality']

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k='all') # K is selecting features
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_) #calculate the score respect to chi2
dfcolumns = pd.DataFrame(X.columns)  #Creating columns
featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
featuresScores.columns = ['Feature', 'Score']
print(featuresScores)


from collections import Counter
print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X,Y)
print(Counter(Y))

out = ['fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
for i in out:
    # sns.boxplot(df[i])
    # plt.show()
    # print(X[i])
    Q1 = X[i].quantile(0.25)
    Q3 = X[i].quantile(0.75)
    IQR = Q3 - Q1
    # print(IQR)
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    # print(upper)
    # print(lower)
    out1 = X[X[i] < lower].values
    out2 = X[X[i] > upper].values
    X[i].replace(out1, lower, inplace=True)
    X[i].replace(out2, upper, inplace=True)
    # sns.boxplot(X[i])
    # plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.3)

func = [rf, dtc]
for item in func:
    item.fit(X_train, y_train)
    y_pred = item.predict(X_test)
    print(item)
    print(accuracy_score(y_test, y_pred))

gb=GradientBoostingClassifier()
rf.fit(X_train,y_train)
y_pred2 = rf.predict(X_test)
print("gb: ", accuracy_score(y_test,y_pred2))