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

df = pd.read_csv("C:/Users/Admin/Desktop/osk/titanic.csv")
le = LabelEncoder()
le.fit(df['Sex'])
df['Sex'] = le.transform(df['Sex'])

m = df
x = m.drop("PassengerId",axis = 1)
x = x.drop("Cabin",axis=1)
x = x.drop("Embarked",axis=1)
x = x.drop("Survived",axis=1)
x = x.drop("Name",axis=1)
x = x.drop("Ticket",axis=1)

y =m["Survived"]


x['Age'].fillna((df['Age'].mean()),inplace=True)

mod = ExtraTreesClassifier()
mod.fit(x,y)
print(mod.feature_importances_)

feat_importance=pd.Series(mod.feature_importances_,index=x.columns)
feat_importance.nlargest(6).plot(kind="barh") #barh means bar graph h means horizontal
plt.show()

print(Counter(y))
from imblearn.over_sampling import RandomOverSampler
ros  = RandomOverSampler(random_state=0)
x,y = ros.fit_resample(x,y)
print(Counter(y))

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



