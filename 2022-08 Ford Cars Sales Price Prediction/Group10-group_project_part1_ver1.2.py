#22rd Aug 2022
#BU MSBA2022 BootCamp Group Project part1
#Group 10: Jiadai Yu, Yuesen Zhang, Xin Su
#Credit to Prof.Hemant Sangwan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = r'D:\0-Summer Pre\boot camp\Ford-Data-GroupProject.xlsx'
Ford = pd.read_excel(filename , header=0)

#--------------------------------
#Basic Information about the data 

print(Ford.head())
print(Ford.shape)    #shape of data (rows, columns)
print(Ford.dtypes)  # variable types (integer, object, float)
print(Ford.info())  # info about data columns, variable names, etc.
print(Ford.count())  # non-missing observationcounts


#--------------------------------
#Q1: summary statistics

#quick look through all attributes
for i in ['model','year','transmission','fuelType','engineSize']:
    print('prices vary across '+ i +' :')
    print(pd.DataFrame(Ford.groupby(i)['price'].mean()))
    print()

#drop the unrealistic year = 2060 as an input error
Ford.drop(Ford.index[Ford['year'] == 2060], inplace=True)

#'model'
price_by_model = pd.DataFrame(Ford.groupby('model')['price'].mean())
model_count = Ford['model'].value_counts()
model_count_props = Ford['model'].value_counts(normalize = True)
price_by_model_count_srt = pd.concat([price_by_model,model_count,model_count_props] , axis=1).sort_values('price')

print(price_by_model_count_srt)
price_by_model_count_srt.plot(x='price' , y='model' )
plt.show()

#'year'
price_by_year = pd.DataFrame(Ford.groupby('year')['price'].mean())

print(price_by_year.sort_values('price'))
price_by_year.plot()
plt.show()

#'transmission'
price_by_trans = pd.DataFrame(Ford.groupby('transmission')['price'].mean())
print(price_by_trans.sort_values('price'))

#'fuelType'
price_by_fuel = pd.DataFrame(Ford.groupby('fuelType')['price'].mean())
fuelType_count = Ford['fuelType'].value_counts()
print(price_by_fuel.sort_values('price'))
print(fuelType_count)

#'engineSize'
price_by_engineSize = pd.DataFrame(Ford.groupby('engineSize')['price'].mean())
price_by_engineSize.plot()
plt.show()

#--------------------------------
#Q2 identify outlier of price, mileage, tax, and mpg

#import zscore
from scipy.stats import zscore

Ford_outliers = []
for i in ['price', 'mileage', 'tax', 'mpg']:
    standardized = pd.DataFrame(Ford[i].transform(zscore))
    outliers = ((standardized[i] < -3) | (standardized[i] > 3))
    Ford_outliers = Ford[outliers]
    print(i)
    print(Ford[i].agg([np.min,np.max,np.mean,np.median]))
    print("outlier_mean\n",Ford_outliers[i].agg([np.min,np.max,np.mean,np.median]))
print(Ford_outliers.head()) ##Outlier mean
    
#credit to slides-S4 Page18

#price

#mileage

#tax
##outliers: tax=0

#mpg mile per gallon
##outliers: mpg=201.8

#--------------------------------
#Q3 (price, mileage, tax, and mpg) correlation matrix

#price mileage
print(np.corrcoef(Ford['price'],Ford['mileage']))
Ford.plot.scatter(x='price' , y='mileage' )
plt.show()

#price tax
print(np.corrcoef(Ford['price'],Ford['tax']))
Ford.plot.scatter(x='price' , y='tax' )
plt.show()

#price mpg
print(np.corrcoef(Ford['price'],Ford['mpg']))
Ford.plot.scatter(x='price' , y='mpg' )
plt.show()

#mileage tax
print(np.corrcoef(Ford['mileage'],Ford['tax']))
Ford.plot.scatter(x='mileage' , y='tax' )
plt.show()

#mileage mpg
print(np.corrcoef(Ford['mileage'],Ford['mpg']))
Ford.plot.scatter(x='mpg' , y='mileage' )
plt.show()

#tax mpg
print(np.corrcoef(Ford['tax'],Ford['mpg']))
Ford.plot.scatter(x='tax' , y='mpg' )
plt.show()
