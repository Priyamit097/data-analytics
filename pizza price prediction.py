#!/usr/bin/env python
# coding: utf-8

# # import librarys

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # load data set

# In[6]:


data = pd.read_csv("C:\\Users\\PRIYA\\Downloads\\archive (1)\\pizza_v1.csv")


# # find top five rows

# In[118]:


data.head()


# # find top last rows

# In[119]:


data.tail()


# In[9]:


data.shape


# # find statistical problem

# In[10]:


data.describe()


# In[11]:


data.isnull()


# # find null value

# In[12]:


data.isnull().sum()


# In[13]:


data.rename({'price_rupiah':'price'}, axis = 1, inplace = True)


# In[14]:


data.head()


# In[15]:


data['price'] = data["price"].str.replace("Rp","")
data['price'] = data["price"].str.replace(",","").astype('int32')


# In[16]:


data.head()


# In[17]:


def convert (value):
    return value*0.0054


# In[18]:


data['price']=data['price'].apply(convert)


# In[19]:


data.head()


# # data analysis

# ## univariat analysis

# In[20]:


data.columns


# In[21]:


data['company'].value_counts()


# In[22]:


data['price'].value_counts()


# In[23]:


plt.hist(x="price",data=data)
plt.title("price distribution")
plt.show


# In[24]:


plt.hist(x="price")
plt.title("price distribution")
plt.show


# In[25]:


data["diameter"].value_counts()


# In[26]:


sns.countplot(data["diameter"])


# In[27]:


data["topping"].value_counts()


# In[28]:


sns.countplot(y = data["topping"])


# In[29]:


data["variant"].value_counts()


# In[30]:


sns.countplot(y = data["variant"])


# In[31]:


data["size"].value_counts()


# In[32]:


sns.countplot(y = data["size"])


# In[33]:


data.head()


# In[34]:


sns.countplot(x = data["extra_sauce"])


# In[35]:


sns.countplot(x = data["extra_cheese"])


# # bivariate analysis

# ## price by company

# In[36]:


data.columns


# In[37]:


sns.barplot(data["company"],data["price"])


# ## price by topping

# In[38]:


sns.boxplot(y = "topping",x = "price" , data=data)


# ## price by size

# In[39]:


sns.boxplot(y = "price",x = "size",data=data)


# # fine the most expensive pizza

# In[40]:


data.columns


# In[41]:


data[data["price"].max()==data["price"]]


# ## find te diameter jambu size pizza

# In[42]:


data[data['size']=='jumbo']['diameter'].head()


# ## find the XL size pizza

# In[43]:


data[data['size']=='XL']['diameter'].head()


# In[44]:


data[(data['size']=='jumbo') & (data['diameter']<=16)]


# In[45]:


data = data.drop(data.index[[6,11,16,80]])


# In[46]:


data[(data['size']=='jumbo') & (data['diameter']<=16)]


# # lable encoding

# In[47]:


cat_cols = data.select_dtypes(include=['object']).columns
print(cat_cols)


# In[48]:


data.head()


# In[49]:


from sklearn.preprocessing import LabelEncoder


# In[50]:


en=LabelEncoder()
for i in cat_cols:
    data[i]=en.fit_transform(data[i])


# In[51]:


data.head()


# ## feature matrix in X response(targe) in vectore y

# In[52]:


x = data.drop('price',axis = 1)
y = data['price']


# ## spliting the dataset train and test set

# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=30,random_state=1)


# In[55]:


x_train


# In[56]:


x_test.shape


# In[57]:


x_train.shape


# In[58]:


x_test


# In[59]:


y_train.shape


# In[60]:


y_test.shape


# ## import the model

# In[61]:


get_ipython().system('pip install xgboost')


# In[62]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# # model traning

# In[104]:


lr = LinearRegression()
lr.fit(x_train,y_train)

svm = SVR()
svm.fit(x_train,y_train)

rm = RandomForestRegressor()
rm.fit(x_train,y_train)

gb = GradientBoostingRegressor()
gb.fit(x_train,y_train)

xg = XGBRegressor()
xg.fit(x_train,y_train)


# # prediction on test data

# In[64]:


y_pred1 = lr.predict(x_test)
y_pred2 = svm.predict(x_test)
y_pred3 = rm.predict(x_test)
y_pred4 = gb.predict(x_test)
y_pred5 = xg.predict(x_test)


# # Evaluating the algorithms

# In[65]:


from sklearn import metrics


# In[66]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)
score5 = metrics.r2_score(y_test,y_pred5)


# In[67]:


print(score1,score2,score3,score4,score5)


# In[68]:


final_data = pd.DataFrame({'models':['lr','svm','rm','gb','xg'],'R2_score':[score1,score2,score3,score4,score5]})


# In[69]:


print(final_data)


# In[70]:


sns.barplot(final_data['models'],final_data['R2_score'])


# # feature importance

# In[71]:


rm.feature_importances_


# In[72]:


fea = pd.Series(rm.feature_importances_,index = x_train.columns)


# In[74]:


fea.plot(kind='barh')


# In[75]:


gb.feature_importances_


# In[84]:


gea = pd.Series(gb.feature_importances_,index = x_train.columns)


# In[85]:


gea.plot(kind='barh')


# In[78]:


xg.feature_importances_


# In[82]:


xea = pd.Series(xg.feature_importances_,index = x_train.columns)


# In[83]:


xea.plot(kind='barh')


# # save the models

# In[100]:


X = data.drop('price',axis=1)
Y = data['price']


# In[105]:


xg = XGBRegressor()


# In[106]:


xg.fit(X,Y)


# In[107]:


import joblib


# In[109]:


joblib.dump(xg,'pizza price pridiction')


# In[111]:


model = joblib.load('pizza price pridiction')


# In[113]:


df = pd.DataFrame({
    'company':1,
    'diameter':22.0,
    'topping':2,
    'variant':8,
    'size':1,
    'extra_sauce':1,
    'extra_cheese':1
},index=[0])


# In[114]:


df


# In[115]:


model.predict(df)


# # GUI

# In[116]:


from tkinter import *
import joblib
import pandas as pd


# In[117]:


def show_entry():
    
    p1= float(e1.get()
    p2= float(e2.get()
    p3= float(e3.get()
    p4= float(e4.get()
    p5= float(e5.get()              
    p6= float(e6.get()
    p7= float(e7.get()
    p8= float(e8.get()
              
    model = joblib.load('pizza price pridiction') 
    df = pd.DataFrame({
    'company':1,
    'diameter':22.0,
    'topping':2,
    'variant':8,
    'size':1,
    'extra_sauce':1,
    'extra_cheese':1
     },index=[0])


# In[ ]:




