import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

data_train=pd.read_csv('D:/CSV Files/Network intrusion detection/Train_data.csv')
data_test=pd.read_csv('D:/CSV Files/Network intrusion detection/Test_data.csv')

data_train.head(5)

data_train.shape

data_train.info()

data_train.describe().T

def data_proflileing(df):
    data_profile = []
    columns = df.columns
    for col in columns:
        dtype = df[col].dtypes
        nunique = df[col].nunique()
        null = df[col].isnull().sum()
        duplicates = df[col].duplicated().sum()
        data_profile.append([col,dtype,nunique,null,duplicates])
    data_profile_finding = pd.DataFrame(data_profile)
    data_profile_finding.columns = ['column','dtype','nunique','null','duplicates']
    return data_profile_finding

data_proflileing(data_train)

data_train.isnull().sum()

for i in data_train.columns:
    print(data_train[i].nunique)

data_train.columns

sns.countplot(data=data_train,x='class',palette='PRGn')

from sklearn import preprocessing

def encoding(df):
    for col in df.columns:
        if df[col].dtype == 'object':
                label_encoder = preprocessing.LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])


encoding(data_train)

data_train.head(2)

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

X = data_train.drop(['class'], axis=1)
y = data_train['class']


mutual_info = mutual_info_classif(X,y)
mutual_info

mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info.sort_values(ascending=False)

select_best_cols = SelectKBest(mutual_info_classif,k=25)
select_best_cols.fit(X,y)
selected_features = X.columns[select_best_cols.get_support()]

X=X[selected_features]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sc = StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve,confusion_matrix


def classalgo_test(x_train, x_test, y_train, y_test):  # classification

    g = GaussianNB()
    b = BernoulliNB()
    kc = KNeighborsClassifier()
    lr = LogisticRegression()
    dc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    gbc = GradientBoostingClassifier()
    xgb = XGBClassifier()
    Bagging = BaggingClassifier()
    AdaBoost = AdaBoostClassifier()

    algos = [g, b, kc, lr, dc, rfc, gbc, xgb, Bagging, AdaBoost]
    algo_names = ['GaussianNB', 'BernoulliNB', 'KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier',
                  'RandomForestClassifier', 'GradientBoostingClassifier', 'BaggingClassifier', 'XGBClassifier',
                  'AdaBoostClassifier']
    Train_acc = []
    Train_precsc = []
    Train_fsc = []
    Train_Recall = []
    Test_acc = []
    Test_precsc = []
    Test_fsc = []
    Test_Recall = []
    Test_AUC = []

    result = pd.DataFrame(index=algo_names)

    for algo in algos:
        algo.fit(x_train, y_train)
        y_train_pred = algo.predict(x_train)
        y_test_pred = algo.predict(x_test)
        Train_acc.append(accuracy_score(y_train, y_train_pred))
        Train_precsc.append(precision_score(y_train, y_train_pred))
        Train_fsc.append(f1_score(y_train, y_train_pred))
        Train_Recall.append(recall_score(y_train, y_train_pred, average='micro'))

        Test_acc.append(accuracy_score(y_test, y_test_pred))
        Test_precsc.append(precision_score(y_test, y_test_pred))
        Test_fsc.append(f1_score(y_test, y_test_pred))
        Test_Recall.append(recall_score(y_test, y_test_pred, average='micro'))
        Test_AUC.append(roc_auc_score(y_test, y_test_pred))

    result['Train_Accuracy Score'] = Train_acc
    result['Train_Precision Score'] = Train_precsc
    result['Train_F1Score'] = Train_fsc
    result['Train_Recall'] = Train_Recall
    result['Test_Accuracy Score'] = Test_acc
    result['Test_Precision Score'] = Test_precsc
    result['Test_F1Score'] = Test_fsc
    result['Test_Recall'] = Test_Recall
    result['Test_AUC_Score'] = Test_AUC

    return result.sort_values('Test_Accuracy Score', ascending=False)

classalgo_test(X_train,X_test,y_train,y_test)


rf=RandomForestClassifier()
rf.fit(X_train,y_train)

rf_predict=rf.predict(X_test)
print(classification_report(y_test,rf_predict))