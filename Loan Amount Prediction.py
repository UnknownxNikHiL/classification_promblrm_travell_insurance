#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import llibraries 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)


# In[2]:


df = pd.read_csv(r"C:\Users\nikhi\OneDrive\Desktop\loan amount\train.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[ ]:





# In[ ]:





# In[9]:


df.isnull().sum()


# In[10]:


sns.distplot(df['Loan Amount Request (USD)'])
plt.show()


# In[11]:


## To check Normal Distribution


sns.distplot(np.log10(df['Loan Amount Request (USD)']))


# In[12]:


sns.distplot(df['Loan Sanction Amount (USD)'])


# In[13]:


df['Income (USD)']=df['Income (USD)'].fillna(2223.135)


# In[14]:


df['Income Stability'].unique()


# In[15]:


df['Income Stability'] = df['Income Stability'].fillna(method='ffill')


# In[16]:


df['Profession'].unique()


# In[17]:


df['Type of Employment'].unique()


# In[18]:


df['Type of Employment'] = df['Type of Employment'].fillna(method='bfill')


# In[19]:


df['Location'].unique()


# In[20]:


df['Current Loan Expenses (USD)'].median()


# In[21]:


df['Current Loan Expenses (USD)'] = df['Current Loan Expenses (USD)'].fillna(374.78)


# In[22]:


df['Income (USD)'].unique()


# In[23]:


df['Dependents'].median()


# In[24]:


df['Dependents'] = df['Dependents'].fillna(2)


# In[25]:


df['Credit Score'].median()


# In[26]:


df['Credit Score']= df['Credit Score'].fillna(739.71)


# In[27]:


df['Has Active Credit Card'].unique()


# In[28]:


df['Has Active Credit Card'] = df['Has Active Credit Card'].fillna(method='ffill')


# In[29]:


df['Has Active Credit Card']= df['Has Active Credit Card'].fillna('Active')


# In[30]:


df['Property Age'].median()


# In[31]:


df['Property Age']= df['Property Age'].fillna(2222.37)


# In[32]:


df['Property Location'].unique()


# In[33]:


df['Property Location'] = df['Property Location'].fillna(method='bfill')


# In[34]:


df['Loan Sanction Amount (USD)'].median()


# In[35]:


df['Loan Sanction Amount (USD)'] = df['Loan Sanction Amount (USD)'].fillna(35209)


# In[36]:


df.isnull().sum()


# In[37]:


plt.figure(figsize=(15,5), dpi=100)
sns.countplot(x = 'Profession',data = df,palette='viridis')
plt.show()


# In[38]:


plt.figure(figsize=(30,5), dpi=100)
sns.countplot(x = 'Type of Employment',data = df,palette='viridis')
plt.show()


# ## Profession wise income

# In[39]:


plt.figure(figsize=(30,8))
sns.barplot(x='Profession', y='Income (USD)', data=df)
plt.show()


# ## Type of Employment wise income

# In[40]:


plt.figure(figsize=(30,8))
sns.barplot(x='Type of Employment', y='Income (USD)', data=df)
plt.show()


# # Employment wise credit score

# In[41]:


plt.figure(figsize=(35,8))
sns.barplot(x='Type of Employment', y='Credit Score', data=df)
plt.show()


# # Location wise credit score

# In[42]:


plt.figure(figsize=(25,8))
sns.barplot(x='Location', y='Credit Score', data=df)
plt.show()


# ## Profession Wise Loan Amount Sanction

# In[43]:


plt.figure(figsize=(25,8))
sns.barplot(x='Profession', y='Loan Sanction Amount (USD)', data=df)
plt.show()


# ## Type of Employement wise Loan Amount Sanction

# In[44]:


plt.figure(figsize=(35,8))
sns.barplot(x='Type of Employment', y='Loan Sanction Amount (USD)', data=df)
plt.show()


# In[45]:


# to check co-releation 
plt.figure(figsize=(15,15), dpi=100)
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


df.dtypes


# In[47]:


df[['Income Stability','Profession','Type of Employment','Location','Expense Type 1','Expense Type 2','Has Active Credit Card','Property Location']].nunique()


# ## Drpo Unwanted Columns

# In[48]:


df = df.drop(['Customer ID', 'Name', 'Property ID'], axis=1)


# ## Categorical to Numeric

# In[49]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[50]:


df['Income Stability'] = label_encoder.fit_transform(df['Income Stability'])
df['Profession'] = label_encoder.fit_transform(df['Profession'])


# # One Hot-Encoding

# In[51]:


df = pd.get_dummies(df,drop_first=True)


# # Sacling

# In[52]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler       ## Robus scaling is use fro Handling the OUTLIERS



# In[53]:


columns_to_plot = ['Age', 'Income (USD)', 'Income Stability', 'Profession',
       'Loan Amount Request (USD)', 'Current Loan Expenses (USD)',
       'Dependents', 'Credit Score', 'No. of Defaults',
       'Property Age', 'Property Type', 'Co-Applicant', 'Property Price',
       'Loan Sanction Amount (USD)']

# Create subplots for each box plot
plt.figure(figsize=(14, 10))  # Adjust the figure size as needed

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(4, 4, i)  # Create a 4x4 grid of subplots
    sns.boxplot(data=df, y=column)
    plt.title(f'Box Plot for {column}')
    plt.xlabel(column)
    plt.ylabel('Values')

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()


# In[ ]:





# In[54]:


x = df.drop('Loan Sanction Amount (USD)', axis=1)
y = df['Loan Sanction Amount (USD)']


# In[55]:


from sklearn.preprocessing import RobustScaler
rs = RobustScaler()


# In[56]:


from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
rs.fit(x)
rs_x = rs.transform(x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[57]:


df.columns


# ( transform = ColumnTransformer(transformers=[
#     ('c1', RobustScaler(),['Income (USD)']),
#     ('c2',RobustScaler(),['Income Stability']),
#     ('c3',RobustScaler(),['Loan Amount Request (USD)']),
#     ('c4',RobustScaler(),['Current Loan Expenses (USD)']),
#     ('c5',RobustScaler(),['Dependents']),
#     ('c6',RobustScaler(),['No. of Defaults']),
#     ('c7',RobustScaler(),['Property Age']),
#     ('c8',RobustScaler(),['Property Price']),
#     ('c9',RobustScaler(),['Co-Applicant']),
# ],remainder = 'passthrough'))

# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


x_train, x_test, y_train, y_test = train_test_split(rs_x, y, test_size=0.30, random_state=42)


# In[ ]:





# In[60]:


#x_train_scl = rs.fit_transform(x_train)
#x_test_scl = rs.fit_transform(x_test)


# # Linear Regression 

# In[61]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[62]:


lr_mdl = lr.fit(x_train, y_train)
lr_mdl


# In[63]:


lr_y_pred = lr_mdl.predict(x_test)
lr_y_pred 


# In[64]:


from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, mean_squared_error, adjusted_rand_score 


# In[ ]:





# In[65]:


r2 = r2_score(y_test, lr_y_pred)
r2


# In[66]:


RMSE = np.sqrt(mean_squared_error(y_test,lr_y_pred))
RMSE


# In[67]:


mae = mean_absolute_error(y_test, lr_y_pred)
mae


# In[68]:


mse = mean_squared_error(y_test, lr_y_pred)
mse


# # Hyperparameter tunning with Lasso & Ridge

# ### Lasso 

# In[69]:


from sklearn.linear_model import Lasso


# In[70]:


lasso = Lasso()


# In[71]:


lasso_mdl = lasso.fit(x_train, y_train)
lasso_mdl


# In[72]:


lasso_y_pred = lasso_mdl.predict(x_test)
lasso_y_pred


# In[73]:


r2 = r2_score(y_test, lasso_y_pred)
r2


# ### Ridge

# In[74]:


from sklearn.linear_model import Ridge
ridge = Ridge()


# In[75]:


ridge_mdl = ridge.fit(x_train, y_train)
ridge_mdl


# In[76]:


ridge_y_pred = ridge_mdl.predict(x_test)
ridge_y_pred


# In[77]:


r2 = r2_score(y_test, ridge_y_pred)
r2


# # SVM Regression

# In[78]:


from sklearn.svm import SVR
svm = SVR()


# In[79]:


svm_mdl = svm.fit(x_test, y_test)
svm_mdl


# In[80]:


svm_y_pred = svm_mdl.predict(x_test)
svm_y_pred


# In[81]:


r2 = r2_score(y_test, svm_y_pred)
r2


# In[ ]:





# In[ ]:





# In[ ]:





# # K-NN Regressor 

# In[82]:


from sklearn.neighbors import KNeighborsRegressor


# In[83]:


knn = KNeighborsRegressor()


# In[84]:


knn_mdl = knn.fit(x_train, y_train)
knn_mdl


# In[85]:


knn_y_pred = knn_mdl.predict(x_test)
knn_y_pred


# In[86]:


r2 = r2_score(y_test, knn_y_pred)
r2


# In[ ]:





# # Decision Tree

# In[87]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()


# In[88]:


dt_mdl = dt.fit(x_train, y_train)
dt_mdl


# In[89]:


dt_y_pred = dt_mdl.predict(x_test)
dt_y_pred


# In[90]:


r2 = r2_score(y_test, dt_y_pred)
r2


# In[ ]:





# In[ ]:





# #### TUNNing

# In[ ]:





# In[91]:


DecisionTreeRegressor()

# Define hyperparameters and their values for tuning
param_grid = {
   'max_depth': [None, 5, 10, 15, 20, 30],
   'min_samples_split': [2, 5, 10, 15],
   'min_samples_leaf': [1, 2, 4, 6, 8, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_max_depth = grid_search.best_params_['max_depth']
best_min_samples_split = grid_search.best_params_['min_samples_split']
best_min_samples_leaf = grid_search.best_params_['min_samples_leaf']

# Train a final DecisionTreeRegressor model with the best hyperparameters
final_dt = DecisionTreeRegressor(max_depth=best_max_depth, 
                                            min_samples_split=best_min_samples_split, 
                                            min_samples_leaf=best_min_samples_leaf)
final_dt.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = final_dt.predict(x_test)

# Calculate Mean Squared Error (MSE) to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Best max_depth: {best_max_depth}")
print(f"Best min_samples_split: {best_min_samples_split}")
print(f"Best min_samples_leaf: {best_min_samples_leaf}")
print(f"Mean Squared Error: {mse}")


# In[ ]:





# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[ ]:


rf_mdl = rf.fit(x_train, y_train)
rf_mdl


# In[ ]:


rf_y_pred = rf_mdl.predict(x_test)
rf_y_pred


# In[ ]:


r2 = r2_score(y_test, rf_y_pred)
r2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Create a RandomForestRegressor
rf = RandomForestRegressor()

# Define hyperparameters and their values for tuning
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_n_estimators = grid_search.best_params_['n_estimators']
best_max_features = grid_search.best_params_['max_features']
best_max_depth = grid_search.best_params_['max_depth']
best_min_samples_split = grid_search.best_params_['min_samples_split']
best_min_samples_leaf = grid_search.best_params_['min_samples_leaf']
best_bootstrap = grid_search.best_params_['bootstrap']

# Train a final RandomForestRegressor model with the best hyperparameters
final_rf = RandomForestRegressor(n_estimators=best_n_estimators, 
                                           max_features=best_max_features, 
                                           max_depth=best_max_depth, 
                                           min_samples_split=best_min_samples_split, 
                                           min_samples_leaf=best_min_samples_leaf, 
                                           bootstrap=best_bootstrap)
final_rf.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = final_rf.predict(x_test)

# Calculate Mean Squared Error (MSE) to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Best n_estimators: {best_n_estimators}")
print(f"Best max_features: {best_max_features}")
print(f"Best max_depth: {best_max_depth}")
print(f"Best min_samples_split: {best_min_samples_split}")
print(f"Best min_samples_leaf: {best_min_samples_leaf}")
print(f"Best bootstrap: {best_bootstrap}")
print(f"Mean Squared Error: {mse}")


# In[ ]:




