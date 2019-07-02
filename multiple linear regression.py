import  numpy as np
import matplotlib.pyplot  as  plt
import pandas as pd 

#Get the dataset

dataset = pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#one hot encoding  the categorical variables 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()   #labelencoder object
X[:, 3]=labelencoder_X.fit_transform(X[:,3])    #fitting the index on it
onehotencoder= OneHotEncoder(categorical_features= [3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X=X[:,1:]

###training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size= 0.2,random_state=0 )

#fitting the multiple linear regreesion model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test results
y_pred= regressor.predict(X_test)


#Bulding the optimal model for backward elimination
import statsmodels.formula.api  as sm
X=np.append(arr= np.ones((50,1)).astype(int) ,values =X ,axis= 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS =sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()


# remove p values that are greater than the significant value ,( my significant value is 5%)

X_opt = X[:, [0,1,2,4,5]]
regressor_OLS =sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

##################
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS =sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()
##############
X_opt = X[:, [0,3,4,5]]
regressor_OLS =sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

####################

##################
X_opt = X[:, [0,3,5]]
regressor_OLS =sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary() 

#####################
X_opt = X[:, [0,3]]
regressor_OLS =sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary() 
