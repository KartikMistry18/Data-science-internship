
import pandas as pd
import sklearn
df = pd.read_csv("D:/CSV Files/housing.csv")
print(df)

df = pd.read_csv("D:/CSV Files/housing.csv")
print(df)
print(df.head(10))
print(df.tail())
print(df.columns.values)
print(df.describe())


#Preparing X and Y
X = df.drop('Id', axis = 1)
X = X.drop('Species', axis = 1)
Y = df['Species']
print(X)
print(Y)