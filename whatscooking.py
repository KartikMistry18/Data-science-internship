
#df = pd.read_json("D:/CSV Files/Whats Cooking/train.json")
#testset = pd.read_json("D:/CSV Files/Whats Cooking/test.json")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble  import RandomForestClassifier

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression()
nb=MultinomialNB()
dt=DecisionTreeClassifier(random_state=0)

d_c = ["moroccan","korean","japanese","vietnamese","brazilian","southern_us","british","cajun_creole","chinese","filipino","french","greek","indian","spanish","irish","russian","noroccan","italian","thai","mexican","jamaican"]
df = pd.read_json("D:/CSV Files/Whats Cooking/train.json")

print(df)

print(df["cuisine"].unique())
x=df['ingredients']
y=df['cuisine'].apply(d_c.index)

df['all_ingredients']=df['ingredients'].map(";".join)
cv=CountVectorizer()
x=cv.fit_transform(df['all_ingredients'].values)

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

rf.fit(X_train,y_train)
r_pred=rf.predict(X_test)

print("Random Forest:",accuracy_score(y_test,r_pred))