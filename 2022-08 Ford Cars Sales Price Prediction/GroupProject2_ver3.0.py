#Date 2022-08-27
#Group 10: Jiadai Yu(U63347232), Yuesen Zhang(U25148433), Xin Su(U07869307)
#Credit to Linear Regression-Housing Data-Python code 

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import warnings

filename = r'D:\0-Summer Pre\boot camp\group project\Ford-Data-GroupProject.xlsx'
Ford = pd.read_excel(filename , header=0)


# In[2]:


#--------------------------------
#Basic Information about the data 

print(Ford.head())
print(Ford.shape)
print(Ford.info())


# In[3]:


#drop the unrealistic year = 2060 as an input error

Ford.drop(Ford.index[Ford['year'] > 2022], inplace=True)


# In[4]:


#categorical variables:

for i in ['model','transmission','fuelType','tax']:
    print('prices vary across '+ i +' :')
    price_by_i = pd.DataFrame(Ford.groupby(i)['price'].mean())
    i_count = Ford[i].value_counts()
    price_by_i_count_srt = pd.concat([price_by_i,i_count] , axis=1).sort_values('price')
    print(price_by_i_count_srt)
    print('\n')


# In[5]:


#how categorical variables attribute to price

fig=plt.figure(figsize=(10,4))
sns.barplot(x = Ford.transmission, y = Ford.price)

fig=plt.figure(figsize=(10,4))
sns.barplot(x = Ford.fuelType, y = Ford.price)


# In[6]:


#mean price of Semi-Auto and Automatic is similar, divide the transmission of cars into 'manual' and assign as 1, 'not manual' assign as 2
#similarly assign fuelType as 'not Hybrid' and 'Hybrid'

##mapping, and assign new columns

trans_mapping = {'Manual':1,'Semi-Auto':2,'Automatic':2}
Ford['trans_num'] = Ford['transmission'].replace(trans_mapping)

fuel_mapping = {'Petrol':1,'Diesel':1,'Other':1,'Electric':1,'Hybrid':2}
Ford['fuel_num'] = Ford['fuelType'].replace(fuel_mapping)

#Merge, to reduce the interference of categorical data to the regression


# In[7]:


#how numeric variables attribute to price

for i in ['year','trans_num','fuel_num','engineSize','mileage','tax','mpg']:
    x = Ford[i]
    y = Ford['price']
    print(np.corrcoef(Ford[i],Ford['price']))
    plt.scatter(x, y, marker='o')
    plt.title(i)
    plt.xlabel(i)
    plt.ylabel('price')
    plt.show()


# In[8]:


#most engineSize fall near 0 and between 1-3
#most tax fall near 0 and between 100-350
#most mpg fall between 10-90, with a few high outliers

#year & mileage seemed to be correlated with log(price)
Ford['log_price'] = np.log(Ford['price'])

for i in ['year','mileage']:
    x = Ford[i]
    y = Ford['log_price']
    print(np.corrcoef(Ford[i],Ford['log_price']))
    plt.scatter(x, y, marker='o')
    plt.title(i)
    plt.xlabel(i)
    plt.ylabel('log_price')
    plt.show()


# In[9]:


# Correlation Matrix - states the relation between each column
correlation_matrix = Ford.corr().round(2)
# annot = True to print the values inside the square
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(data=correlation_matrix, annot=True, ax = ax)


# In[10]:


#Model 1

#year has 0.65 correlation with price, mileage has -0.53 correlation with price,
#meanwhile year and mileage have -0.72 correlation with each other, are highly dependent

#engineSize and tax both have 0.41 correlation with price
#tax have -0.5 correlation with mpg

#Model1: X = year, mileage, engineSize, tax, Y = price


# In[11]:


#Model 2

#year has 0.77 correlation with log_price, mileage has -0.63 correlation with log_price,
#meanwhile year and mileage have -0.72 correlation with each other, are highly dependent

#tax 0.38,and tax has -0.5 correlation with mpg
# engineSize 0.29

#Model2: X = year, mileage, engineSize, tax, Y = log_price


# In[12]:


# converting the daya set into array
X1 = np.array(pd.concat([Ford['year'],Ford['mileage'],Ford['engineSize'],Ford['tax']],axis=1))
Y1 = Ford['price'].values
X2 = np.array(pd.concat([Ford['year'],Ford['mileage'],Ford['engineSize'],Ford['tax']],axis=1))
Y2 = Ford['log_price'].values

# splitting the dataset into train and test. 30% of the whole data is for testing
x1_train, x1_test, y1_train, y1_test = train_test_split(X1,Y1, test_size=0.3, random_state = 42)
x2_train, x2_test, y2_train, y2_test = train_test_split(X2,Y2, test_size=0.3, random_state = 42)

print(x1_train == x2_train)


# In[13]:


#Apllying Multiple linear regression model on the dataset
Ford_model1 = LinearRegression()
Ford_model1.fit(x1_train, y1_train)
Ford_model2 = LinearRegression()
Ford_model2.fit(x2_train, y2_train)

#Predicting the train data using the linear model
y1_pred_train = Ford_model1.predict(x1_train)
y2_pred_train = Ford_model2.predict(x2_train)

#Predicting the test data using the Linear model
y1_pred_test = Ford_model1.predict(x1_test)
y2_pred_test = Ford_model2.predict(x2_test)

print("Coefficients of the Linear Regression Model 1:", Ford_model1.coef_)
print("Intercept of the Linear Regression Model 1:", Ford_model1.intercept_)
print("Coefficients of the Linear Regression Model 2:", Ford_model2.coef_)
print("Intercept of the Linear Regression Model 2:", Ford_model2.intercept_)


# In[14]:


#Formula of the model1:
#price = 1.21292845e+03*year - 6.04409697e-02*mileage + 5.67107198e+03*engineSize + 6.50246287e+00*tax - 2441011.0945755113

#Formula of the model2:
#log_price = 1.30621388e-01*year - 4.95493934e-06*mileage + 3.92918890e-01*engineSize + 2.47154619e-04*tax - 254.5480523731459


# In[15]:


# Evaluating them odel using R^2 and Mean Square Error
print("Evaluation of Model 1:")
print("Train Data:")
print("R2:", metrics.r2_score(y1_train, y1_pred_train))
print('MSE:', metrics.mean_squared_error(y1_train, y1_pred_train))
print("Test Data:")
print("R2:", metrics.r2_score(y1_test, y1_pred_test))
print('MSE:', metrics.mean_squared_error(y1_test, y1_pred_test))
print()
print("Evaluation of Model 2:")
print("Train Data:")
print("R2:", metrics.r2_score(y2_train, y2_pred_train))
print('MSE:', metrics.mean_squared_error(y2_train, y2_pred_train))
print("Test Data:")
print("R2:", metrics.r2_score(y2_test, y2_pred_test))
print('MSE:', metrics.mean_squared_error(y2_test, y2_pred_test))


# In[16]:


model1 = sns.regplot(y1_test,y1_pred_test,label='price')
model1.legend(loc="best")
plt.show()
model2 = sns.regplot(y2_test,y2_pred_test,label='log_price',color='g')
model2.legend(loc="best")
plt.show()


# In[17]:


# Viewing the Actual and Predicted test data
y1_pred_test_prop = 100 * (y1_pred_test - y1_test) / y1_test
y2_pred_test_prop = 100 * ( np.exp(y2_pred_test) - np.exp(y2_test)) / np.exp(y2_test)

pd.DataFrame({"Actual price1 $": y1_test,"Model1 Predict $": y1_pred_test, 'Percentage difference1 %':y1_pred_test_prop,              "Actual price2 $": np.exp(y2_test),"Model2 Predict $": np.exp(y2_pred_test), 'Percentage difference2 %':y2_pred_test_prop})


# In[ ]:




