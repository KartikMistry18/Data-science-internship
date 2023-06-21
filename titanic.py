
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import numpy
df = pd.read_csv("D:/CSV Files/titanic_data.csv")
print(df)

df = pd.read_csv("D:/CSV Files/titanic_data.csv")
print(df)
print(df.head(10))
print(df.tail())
print(df.columns.values)
print(df.describe())


###############################################################
#preparing X & Y
X= df.drop('Survived',axis=1)
X= X.drop('Embarked',axis=1)
Y=df['Embarked']
print(X)
print(Y)

###############################################################
'''
# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures=SelectKBest(score_func=chi2,k='all')
fit = bestfeatures.fit(X,Y)                                      #training the model
dfscores = pd.DataFrame(fit.scores_)                             #storing the scores
dfcolumns = pd.DataFrame(X.columns)                              #storing coloumns
featuresScores = pd.concat([dfcolumns,dfscores],axis=1)          #concat scores and coloumns
featuresScores.columns = ['Survived','Scores']                   #giving lables to scores and columns

print(featuresScores)
'''
##############################################################################33
#Numerical to Categorical
df['Age']=pd.cut(df['Age'],2,labels=['0','1'])
df['PassengerId']=pd.cut(df['PassengerId'],2,labels=['0','1'])
df['Survived']=pd.cut(df['Survived'],2,labels=['0','1'])
df['Pclass']=pd.cut(df['Pclass'],2,labels=['0','1'])
df['SibSp']=pd.cut(df['SibSp'],2,labels=['0','1'])
df['Parch']=pd.cut(df['Parch'],2,labels=['0','1'])
df['Fare']=pd.cut(df['Fare'],2,labels=['0','1'])

print(df)
################################################################################

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

le.fit(df['Age'])
df['Age']=le.transform(df['Age'])

le.fit(df['PassengerId'])
df['PassengerId']=le.transform(df['PassengerId'])

le.fit(df['Survived'])
df['Survived']=le.transform(df['Survived'])

le.fit(df['Pclass'])
df['Pclass']=le.transform(df['Pclass'])

le.fit(df['SibSp'])
df['SibSp']=le.transform(df['SibSp'])

le.fit(df['Parch'])
df['Parch']=le.transform(df['Parch'])

le.fit(df['Fare'])
df['Fare']=le.transform(df['Fare'])

######################################################

#Dealing with null values
print(df)
print("Number of null values")
print(df.isnull().sum)
print(df.isnull().sum())

print("Number of not null")
print(df.notnull().sum)
print(df.notnull().sum())

m = df
x = m.drop("PassengerId",axis = 1)
x = x.drop("Cabin",axis=1)
x = x.drop("Embarked",axis=1)
x = x.drop("Survived",axis=1)
x = x.drop("Name",axis=1)
x = x.drop("Ticket",axis=1)

y =m["Survived"]


x['Age'].fillna((df['Age'].mean()),inplace=True)

#######################################################

mod = ExtraTreesClassifier()
mod.fit(x, y)
print(mod.feature_importances_)

feat_importance=pd.Series(mod.feature_importances_,index=x.columns)
feat_importance.nlargest(6).plot(kind="barh") #barh means bar graph h means horizontal
plt.show()

########################################################

print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
ros  = RandomOverSampler(random_state=0)
x, y = ros.fit_resample(x, y)
print(Counter(y))

#########################################################

logr = LogisticRegression()
rt=RandomForestClassifier()
pca = PCA(n_components=2)

pca.fit(x)
x=pca.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=10,test_size=0.1)

logr.fit(x_train,y_train)
y_pred = logr.predict(x_test)

print(accuracy_score(y_test,y_pred))

rt.fit(x_train,y_train)
y_pred1 = rt.predict(x_test)
print("RT: ", accuracy_score(y_test,y_pred1))


