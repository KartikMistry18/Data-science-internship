
# manipulation data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
#visualiation data
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


#default theme
plt.style.use('ggplot')
sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'


train = pd.read_csv('D:/CSV Files/blackfriday2.csv')
test = pd.read_csv('D:/CSV Files/blackfriday.csv')
train.head(5)

train.shape
train.info()

# show the numerical values

num_columns = [f for f in train.columns if train.dtypes[f] != 'object']
num_columns.remove('Purchase')
num_columns.remove('User_ID')
num_columns

# show the categorical values

cat_columns = [f for f in train.columns if train.dtypes[f] == 'object']
cat_columns

train.describe(include='all')

# Finding the missing values

missing_values=train.isnull().sum()
percent_missing = train.isnull().sum()/train.shape[0]*100

value = {
    'missing_values':missing_values,
    'percent_missing':percent_missing
}
frame=pd.DataFrame(value)
frame

# Product Category 2
train.Product_Category_2.value_counts()
train.Product_Category_2.describe()

# Replace using median

median = train['Product_Category_2'].median()
train['Product_Category_2'].fillna(median, inplace=True)

# Product Category 3
train.Product_Category_3.value_counts()

# drop Product_Category_3
train=train.drop('Product_Category_3',axis=1)


missing_values=train.isnull().sum()
percent_missing = train.isnull().sum()/train.shape[0]*100

value = {
    'missing_values':missing_values,
    'percent_missing':percent_missing
}
frame=pd.DataFrame(value)
frame

train.hist(edgecolor='black',figsize=(12,12));

train.columns

#Drop userId and ProductId
train = train.drop(['Product_ID','User_ID'],axis=1)

# checking the new shape of data
print(train.shape)
train

# Label Encoding
df_Gender = pd.get_dummies(train['Gender'])
df_Age = pd.get_dummies(train['Age'])
df_City_Category = pd.get_dummies(train['City_Category'])
df_Stay_In_Current_City_Years = pd.get_dummies(train['Stay_In_Current_City_Years'])

data_final= pd.concat([train, df_Gender, df_Age, df_City_Category, df_Stay_In_Current_City_Years], axis=1)

data_final.head()

data_final = data_final.drop(['Gender','Age','City_Category','Stay_In_Current_City_Years'],axis=1)
data_final

data_final.dtypes

#splitting the data

from sklearn.model_selection import train_test_split


x=data_final.drop('Purchase',axis=1)
y=data_final.Purchase

print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Linear Regression
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)
print(lm.fit(x_train, y_train))

#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

print('Intercept parameter:', lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
print(coeff_df)

predictions = lm.predict(x_test)
print("Predicted purchases (in dollars) for new costumers:", predictions)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))


