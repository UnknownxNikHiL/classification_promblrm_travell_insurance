#!/usr/bin/env python
# coding: utf-8

# # Travel Insurance

# In[1]:


# Import libraries 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Import Data


df = pd.read_excel(r"C:\Users\nikhi\OneDrive\Desktop\Travell Insurance Data.xlsx")
df


# In[3]:


# check no. of ROWS & Columns
df.shape


# In[4]:


df.dtypes


# In[5]:


# check NaN values

df.isnull().sum()


# In[6]:


# There is no null Values so we dont have to treat any missing values


# In[7]:


# check unique in columns
df['Agency'].unique()


# In[8]:


df['Agency'].nunique()


# In[9]:


# check unique in columns
df['Agency Type'].unique()


# In[10]:


df['Agency Type'].nunique()


# In[11]:


# check unique in columns
df['Distribution Channel'].unique()


# In[12]:


df['Distribution Channel'].nunique()


# In[13]:


# check unique in columns
df['Product Name'].unique()


# In[14]:


df['Product Name'].nunique()


# In[15]:


# check unique in columns
df['Destination'].unique()


# In[16]:


df['Destination'].nunique()


# ## EDA Part

# In[17]:


# check total number of male female are Claim or not
sns.countplot(x = 'Gender',data = df,hue = 'Claim')
plt.show()


# In[18]:


plt.figure(figsize=(10,5), dpi=100)
sns.countplot(x = 'Agency',data = df,hue = 'Claim',palette='viridis')
plt.show()


# In[19]:


plt.figure(figsize=(5,5), dpi=100)
sns.countplot(x = 'Agency Type',data = df,hue = 'Claim',palette='rainbow')
plt.show()


# In[20]:


plt.figure(figsize=(7,5), dpi=100)
sns.countplot(x = 'Distribution Channel',data = df,hue = 'Claim',palette='crest')
plt.show()


# In[21]:


plt.figure(figsize=(70,40), dpi=100)
sns.countplot(x = 'Product Name',data = df,hue = 'Claim',palette='rainbow')
plt.show()


# In[22]:


#check distributon

plt.figure(figsize=(10,7), dpi=80)
sns.histplot(data=df, x='Age', bins=10)
plt.xlabel('AGE')
plt.title('Hitogram')
plt.show()


# In[23]:


#check distributon

plt.figure(figsize=(10,7), dpi=60)
sns.histplot(data=df, x='Duration', bins=10)
plt.xlabel('Duration')
plt.title('Hitogram')
plt.show()


# In[24]:


#check distributon

plt.figure(figsize=(10,7), dpi=60)
sns.histplot(data=df, x='Net Sales', bins=10)
plt.xlabel('Net Sales')
plt.title('Hitogram')
plt.show()


# In[25]:


# For spredness of data
sns.distplot(df['Age'])
plt.show()


# In[26]:


# For spredness of data
sns.distplot(df['Net Sales'])
plt.show()


# In[27]:


# For spredness of data
sns.distplot(df['Duration'])
plt.show()


# In[28]:


# to check co-releation 
plt.figure(figsize=(7,7), dpi=100)
sns.heatmap(df.corr(), annot=True, cmap='rainbow')
plt.show()


# In[160]:


sns.pairplot(df, hue='Claim', palette=['yellow', 'lime'])
plt.show()



# ## Categorical to Numeric

# In[30]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[31]:


df['Agency'] = label_encoder.fit_transform(df['Agency'])
df['Agency Type'] = label_encoder.fit_transform(df['Agency Type'])
df['Distribution Channel'] = label_encoder.fit_transform(df['Distribution Channel'])
df['Product Name'] = label_encoder.fit_transform(df['Product Name'])
df['Destination'] = label_encoder.fit_transform(df['Destination'])
df['Claim'] = label_encoder.fit_transform(df['Claim'])


# ###### OneHotEncoding

# In[32]:


df = pd.get_dummies(df,drop_first=True)


# In[33]:


# Fro checking Outliers

plt.figure(figsize=(15,8),dpi=100)
sns.boxplot(df)


# In[34]:


from scipy.stats import zscore


# In[35]:


df[(zscore(df['Agency'])>3)|(zscore(df['Agency'])<-3)]


# In[36]:


df[(zscore(df['Agency Type'])>3)|(zscore(df['Agency Type'])<-3)]


# In[37]:


df[(zscore(df['Distribution Channel'])>3)|(zscore(df['Distribution Channel'])<-3)]


# In[38]:


df[(zscore(df['Net Sales'])>3)|(zscore(df['Net Sales'])<-3)]


# In[39]:


df[(zscore(df['Commision (in value)'])>3)|(zscore(df['Commision (in value)'])<-3)]


# In[40]:


df[(zscore(df['Age'])>3)|(zscore(df['Age'])<-3)]


# In[41]:


x = df.drop('Claim', axis=1)
y = df['Claim']


# In[42]:


# Vlues colunt in Target Variable
y.value_counts()


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


# Divide data in Traning and Testing part

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 42)
x_train.shape, x_test.shape


# In[45]:


from imblearn.combine import SMOTETomek    ### import for this libarary for SAMPLING


# In[46]:


smpl = SMOTETomek()


# In[47]:


x_train_os, y_train_os = smpl.fit_resample(x_train, y_train)     ### SAMPLING ON TRANING DATA


# In[48]:


y_train_os.value_counts()


# In[49]:


# Scaling
# For Handling the outliers we use Robust Scaling method


# In[50]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler       ## Robus scaling is use fro Handling the OUTLIERS


# In[51]:


transform = ColumnTransformer(transformers=[
    ('c1', RobustScaler(),['Distribution Channel']),
    ('c2',RobustScaler(),['Net Sales']),
    ('c3',RobustScaler(),['Commision (in value)']),
    ('c4',RobustScaler(),['Age']),
],remainder = 'passthrough')


# In[52]:


# Scaling the data
x_train_scl = transform.fit_transform(x_train_os)
x_test_scl = transform.transform(x_test)


# # Logistic Regression 

# In[53]:


from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()


# In[54]:


lor_mdl = lor.fit(x_train_scl, y_train_os)
lor_mdl


# In[55]:


lor_y_pred = lor_mdl.predict(x_test_scl)
lor_y_pred


# In[56]:


## import this libeary for model evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score


# In[131]:


accuracy_score_lor = accuracy_score(lor_y_pred, y_test)
accuracy_score_lor 


# In[58]:


cm = confusion_matrix(lor_y_pred, y_test)


# In[59]:


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[60]:


report = classification_report(lor_y_pred, y_test)
print(report)


# In[61]:


fpr, tpr, _ = (roc_curve(lor_y_pred, y_test))


# In[62]:


plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# # Decision Tree

# In[63]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[64]:


dt_mdl = dt.fit(x_train_scl, y_train_os)
dt_mdl


# In[65]:


dt_y_pred = dt_mdl.predict(x_test_scl)
dt_y_pred


# In[130]:


accuracy_score_dt = accuracy_score(dt_y_pred, y_test)
accuracy_score_dt 


# In[67]:


cm = confusion_matrix(dt_y_pred, y_test)


# In[68]:


sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[69]:


print(classification_report(dt_y_pred, y_test))


# In[70]:


fpr, tpr, _ = roc_curve(dt_y_pred, y_test)


# In[71]:


plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# ##### Tunning

# In[72]:


from sklearn.model_selection import GridSearchCV
tun_dt = [{'criterion':['gini', 'entropy'], 'max_depth':range(1,10), 'min_samples_split': range(1,30)}]
gs_dt = GridSearchCV(DecisionTreeClassifier(), tun_dt, scoring = 'accuracy', cv = 10 )


# In[73]:


gs_dt.fit(x_train_scl, y_train_os)


# In[74]:


gs_dt.best_params_


# In[141]:


tune_dt = gs_dt.best_score_
tune_dt


# In[ ]:





# # Random Forest

# In[76]:


from sklearn.ensemble import RandomForestClassifier
rmf = RandomForestClassifier()


# In[77]:


rmf_mdl = rmf.fit(x_train_scl, y_train_os)
rmf_mdl


# In[78]:


rmf_y_pred = rmf_mdl.predict(x_test_scl)
rmf_y_pred


# In[129]:


accuracy_score_rmf = accuracy_score(rmf_y_pred, y_test)
accuracy_score_rmf


# In[80]:


cm = confusion_matrix(rmf_y_pred, y_test)


# In[81]:


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[82]:


print(classification_report(rmf_y_pred, y_test))


# In[83]:


fpr, tpr, _ = roc_curve(rmf_y_pred, y_test)


# In[84]:


plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# ##### Tunning

# In[116]:


rf_tun =  [{'criterion':['gini', 'entropy'], 'max_depth':[5,10,15], 'min_samples_split': range(1,30), 'n_estimators':range(1,10)}]
gs_rf = GridSearchCV(RandomForestClassifier(), rf_tun, scoring='accuracy', cv =10)


# In[117]:


gs_rf.fit(x_train_scl, y_train_os)


# In[118]:


gs_rf.best_params_


# In[132]:


tune_rmf = gs_rf.best_score_
tune_rmf


# In[ ]:





# In[ ]:





# # XG Boost

# In[85]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[86]:


xgb_mdl = xgb.fit(x_train_scl, y_train_os)


# In[87]:


xgb_y_pred = xgb_mdl.predict(x_test_scl)


# In[134]:


accuracy_score_xgb = accuracy_score(xgb_y_pred, y_test)
accuracy_score_xgb


# In[89]:


cm = confusion_matrix(xgb_y_pred, y_test)


# In[90]:


sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[91]:


print(classification_report(xgb_y_pred, y_test))


# In[92]:


fpr, tpr, _ = roc_curve(xgb_y_pred, y_test)


# In[93]:


plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# ###### Tunning

# In[94]:


tun_xgb = {'learning_rate':[0.1,0.2,0.3], 'max_depth':range(3,7), 'n_estimators':[100,200,300], 'subsample':[0.8,1.0]}
gs_xgb = GridSearchCV(XGBClassifier(), tun_xgb, scoring = 'accuracy', cv = 5)


# In[95]:


gs_xgb.fit(x_train_scl, y_train_os)


# In[96]:


gs_xgb.best_params_


# In[135]:


tune_xgb = gs_xgb.best_score_
tune_xgb


# # SVM

# In[98]:


from sklearn.svm import SVC
svc = SVC()


# In[99]:


svc_mdl = svc.fit(x_train_scl, y_train_os)
svc_mdl


# In[100]:


svc_y_pred = svc_mdl.predict(x_test_scl)
svc_y_pred


# In[143]:


accuracy_score_svm = accuracy_score(svc_y_pred, y_test)
accuracy_score_svm


# In[102]:


cm = confusion_matrix(svc_y_pred, y_test)


# In[103]:


sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[104]:


print(classification_report(svc_y_pred, y_test))


# In[105]:


fpr, tpr, _ = roc_curve(svc_y_pred, y_test)


# In[106]:


plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# # Naive bayes 

# In[107]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()


# In[108]:


nb_mdl = nb.fit(x_train_scl, y_train_os)
nb_mdl


# In[109]:


nb_y_pred = nb_mdl.predict(x_test_scl)
nb_y_pred


# In[137]:


accuracy_score_nb = accuracy_score(nb_y_pred, y_test)
accuracy_score_nb


# In[111]:


cm = confusion_matrix(nb_y_pred, y_test)


# In[112]:


sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[113]:


print(classification_report(nb_y_pred, y_test))


# In[114]:


fpr, tpr, _ = roc_curve(nb_y_pred, y_test)


# In[115]:


plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# # KNN

# In[120]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[121]:


knn_mdl = nb.fit(x_train_scl, y_train_os)
knn_mdl


# In[122]:


knn_y_pred = nb_mdl.predict(x_test_scl)
knn_y_pred


# In[138]:


accuracy_score_knn = accuracy_score(nb_y_pred, y_test)
accuracy_score_knn


# In[152]:


cm = confusion_matrix(nb_y_pred, y_test)


# In[153]:


sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[154]:


print(classification_report(nb_y_pred, y_test))


# In[155]:


fpr, tpr, _ = roc_curve(nb_y_pred, y_test)


# In[156]:


plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# # Accuracy of All Algo

# In[148]:


models = pd.DataFrame({
    "Model": ["Logistic Regression",
              "Decision Tree",
              "Tune Decision Tree",
              "Random Forest",
              "Tune Random Forest",
              "XG Boost",
              "TuneXG Boost",
              "SVM",
              "Naive Bayes",
              "KNN"],
    "Accuracy score": [accuracy_score_lor,accuracy_score_dt,tune_dt, accuracy_score_rmf, tune_rmf,
                       accuracy_score_xgb,tune_xgb, accuracy_score_svm, accuracy_score_nb, accuracy_score_knn]
})

print(models)


# In[151]:


models.sort_values(by="Accuracy score",ascending= False)


# In[ ]:




