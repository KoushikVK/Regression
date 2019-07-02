
import  numpy as np
import matplotlib.pyplot  as  plt
import pandas as pd 

##### importing data set

dataset = pd.read_csv('Salary_Data.csv')

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y, train_size =2/3,random_state=0 )


#implementing linear regression to training set


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test results

#Create an object Y_pred to predict the salaries of employees depending on their experience , on
#correlating  them  with X_test

# Now compare y_pred and y_test to get predicted results...

y_pred = regressor.predict(X_test) 


#Visualizing the training set

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train) ,color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


#to predict new observations

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,regressor.predict(X_train) ,color='blue')
plt.title('Salary vs Experience (Train set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


