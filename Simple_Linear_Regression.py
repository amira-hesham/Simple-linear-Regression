import numpy as np   # For Matrices Manipulations
import pandas as pd  # For Data frames etc 
import xlrd  # For Excel Files
import os   # For Changing to the Correct Directory 
df= pd.read_excel("Insurance.xls")
print(df)
df
X_train=df.iloc[0:40,0].values 
print(X_train)
y_train=df.iloc[40:,0].values
y_train
X_train=df.iloc[0:40,0].values 

y_train= df.iloc[0:40,1].values
print(y_train)
X_test= df.iloc[40:,0].values
y_test= df.iloc[40:,1].values
print('The Features to be trained on :', X_train.shape )
print('The Labels to be trained on :', y_train.shape )
print('The Features to be tested on :', X_test.shape )
print('The Labels to be tested on :', y_test.shape )
print('X_train type : ',type( X_train), 'X_test type :',type(X_test) ,'y_train type :',type( y_train) , 'y_test type :',type( y_test ))
import matplotlib.pyplot as plt # A Great Package For Plotting and Visualization 
plt.figure(figsize=(10,8))     # The Plotting Window Size 
plt.title(' X vs Y Regression', fontsize=24 , fontstyle='italic')  # Title , Font size and Style 
plt.scatter(X_train ,y_train , alpha =0.5, label='Y')  # NOTE : Make Sure  x_train , y_train  IN Data Type ( Array )
plt.show()   # Showing the Result of plotting 
def MSE (actual, predicted):
    return np.sum((actual-predicted)**2) /len(predicted)
MSE(X_train,y_train)
def Mean (values):
    return np.sum(values)/len(values)
  
def Variance(r):
    return np.sum((r-Mean(r))**2) /len(r)
    
def covariance(x,y):
    return np.sum((x-Mean(x))*(y- Mean(y)) )/len(x)
Variance(X_train)
print(Mean(X_train))
print(Variance(X_train))
print(covariance(X_train,y_train))
def coefficents(x,y):
        b1=covariance(x,y) / Variance(x)
        b0=Mean(y)-b1*Mean(x)
        return b0,b1
def SimpleLinearRegression(x_train,y_train,x_test):
 
    b0,b1=coefficents(x_train,y_train)
    return b1*x_test+b0
predict=SimpleLinearRegression(X_train,y_train,X_test)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.title('PREDICTION VS ACTUAL ', fontsize=24 , fontstyle='italic')
plt.scatter(X_test,y_test)
plt.plot(X_test, predict, linewidth=2.0)  ## REPLACE X AND Y WITH  X_test AND PREDS ARRAYS! 
#plt.show 
