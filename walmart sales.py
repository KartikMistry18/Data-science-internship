# Importing all the libraries

import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)

# Reading the data using pandas dataframe
features = pd.read_csv('D:/CSV Files/Walmart/features.csv')
train = pd.read_csv('D:/CSV Files/Walmart/train.csv')
stores = pd.read_csv('D:/CSV Files/Walmart/stores.csv')
test = pd.read_csv('D:/CSV Files/Walmart/test.csv')
sample_submission = pd.read_csv('D:/CSV Files/Walmart/sampleSubmission.csv')

print(features.head())
print("------------------------------------------------------------\n")
print(stores.head())
print("------------------------------------------------------------\n")
print(train.head())
print("------------------------------------------------------------\n")
print(test.head())
print("------------------------------------------------------------\n")
print(sample_submission.head())

# Finding the number of rowns and columns in dataframe
features.shape, train.shape, stores.shape, test.shape

# Some basic information of differnt column's data type of dataframe
print(features.dtypes)
print("------------------------------------------------------------\n")
print(train.dtypes)
print("------------------------------------------------------------\n")
print(stores.dtypes)
print("------------------------------------------------------------\n")
print(test.dtypes)

feature_store = features.merge(stores, how='inner', on = "Store")
train = train.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])
test = test.merge(feature_store, how='inner', on=['Store','Date','IsHoliday'])

# Another useful step is to facilate the acces to the 'Date' attribute by splitting it into its componenents (i.e. Year, Month and week,day).
train = train.copy()
test = test.copy()

train['Date'] = pd.to_datetime(train['Date'])
train['Year'] = pd.to_datetime(train['Date']).dt.year
train['Month'] = pd.to_datetime(train['Date']).dt.month
#train['Week'] = pd.to_datetime(train['Date']).dt.week
train['Day'] = pd.to_datetime(train['Date']).dt.day
train.replace({'A': 1, 'B': 2,'C':3},inplace=True)

test['Date'] = pd.to_datetime(test['Date'])
test['Year'] = pd.to_datetime(test['Date']).dt.year
test['Month'] = pd.to_datetime(test['Date']).dt.month
#test['Week'] = pd.to_datetime(test['Date']).dt.week
test['Day'] = pd.to_datetime(test['Date']).dt.day
test.replace({'A': 1, 'B': 2,'C':3},inplace=True)

print(train.head())
print("------------------------------------------------------------\n")
print(test.head())

Y_train = train['Weekly_Sales']
targets = Y_train.copy()
train= train.drop(['Weekly_Sales'],axis=1)

# Let's also identify the numeric and categorical columns.
numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train.select_dtypes('object').columns.tolist()

print(numeric_cols)
print("------------------------------------------------------------\n")
print(categorical_cols)

# Check if there is any null value in train dataframe
train.isnull().sum()

# Check if there is any null value test in dataframe
test.isnull().sum()

# Create the imputer
imputer = SimpleImputer(missing_values= np.NaN, strategy='mean')

# Fit the imputer to the numeric columns
imputer.fit(train[numeric_cols])

#Replace all the null values
train[numeric_cols] =imputer.transform(train[numeric_cols])

# Check if there is any null value
train.isnull().sum()

# importing MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Create the scaler
scaler = MinMaxScaler()
# Fit the scaler to the numeric columns
scaler.fit(train[numeric_cols])
# Transform and replace the numeric columns
train[numeric_cols] = scaler.transform(train[numeric_cols])
train[numeric_cols].describe().loc[['min', 'max']]

# 'Date' is irrelevant and Drop it from data.
train= train.drop(['Date'],axis=1)
test = test.drop(['Date'], axis=1)

# Preparing the dataset:
X_train =train[['Store','Dept','IsHoliday','Size','Type','Year']]
X_test = test[['Store', 'Dept','IsHoliday', 'Size', 'Type', 'Year']]

print(X_train.columns)
print(X_test.columns)

# Splitting and training
train_inputs, val_inputs, train_targets, val_targets = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

# importing XGBRegressor
from xgboost import XGBRegressor

# fitting the model
model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=20, max_depth=4)

model.fit(train_inputs,train_targets)

#Let's turn this into a dataframe and visualize the most important features.
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance':model.feature_importances_
}).sort_values('importance', ascending=False)

import seaborn as sns
plt.figure(figsize=(10,6))
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature');

# Make and evaluate predictions:
x_pred = model.predict(train_inputs)
x_pred

# calculating mean_squared_error
def rmse(a, b):
    return mean_squared_error(a, b, squared=False)

rmse(x_pred,train_targets)

x_preds=model.predict(X_test)
x_preds

Final = X_test[['Store', 'Dept']]
test['Weekly_Sales']= x_preds

sample_submission['Weekly_Sales'] = test['Weekly_Sales']
sample_submission.to_csv('submission_2.csv',index=False)

preds1=pd.read_csv('submission_2.csv')
preds1

#################
# fitting the model with Hyperparameter Overfitting
RF = RandomForestRegressor(n_estimators=58, max_depth=27, max_features=6, min_samples_split=3, min_samples_leaf=1)
RF.fit(train_inputs,train_targets)

RF.score(train_inputs, train_targets)

RF.score(val_inputs, val_targets)