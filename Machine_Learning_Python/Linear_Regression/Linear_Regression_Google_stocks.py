################################### LINEAR REGRESSION ###########################################
## GUILHERME CARVALHO PEREIRA
## IMPORTING LIBARIES 
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn import linear_model
import sklearn.preprocessing
import math, quandl
import pandas as pd
import numpy as np
## GETING THE DATE SET FORM QUANDL
df = quandl.get('WIKI/GOOGL')
## PREVIEWING THE DATA
#print(df_raw.head())
## GETING THE FEATURES MORE MEANINGFUL
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#################################################################################################
## DEFINING THE SPECIAL RELATIONSHIP BETWEEN THE FEATURES
## HIGH_LOW_PERCENT (HL_PCT) => THIS FEATURE GIVE US THE PERCENT VOLATILITY OF THE STOCKS
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] *100 
## PERCENT_CHANGE (PCT_CHANGE) => THIS FEATURE GIVE US THE DAILY MOVE OF THE STOCKS
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0
## DEFINING THE DATAFRAME WITH THE SPECIAL RELATIONSHIPS
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
## PREVIEWING THE DATA
#print(df.head())
## DEFINING THE LABEL... THE 'THING' WE WANT TO PREDICT
forecast_col = 'Adj. Close'
## FILLING THE NaN DATA ON THE DATASET
df.fillna(-99999, inplace=True)
## GETING 10% OF THE DATAFRAME AND TRY TO PREDICT
pct_data = 0.01  ##GIVE HOW MUCH OF THE DATASET WE ARE TRYING TO PREDICT
forecast_out = int(math.ceil(pct_data*len(df)))
## ADD THE LABEL TO THE DATAFRAME
df['Label'] = df[forecast_col].shift(-forecast_out)  ##JUST THE COLLUMS 10 DAYS ON THE FUTURE
df.dropna(inplace = `True`)
## PREVIEWING THE DATA
#print(df.head())
## DEFINING X AND y FOR OUR HYPOTHESIS
X = np.array(df.drop(['Label'], 1))
y = np.array(df['Label'])
## SCALING OUR FEATURES
X = preprocessing.scale(X)
## DEVIDING OUR DATA INTO TRAINING AND DATA SET...TESTE SIZE IN %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
## DEFINING OUR LINEAR REGRESSION HYPOTHESIS
hypo = linear_model.LinearRegression()
hypo.fit(X_train, y_train)
accuracy = hypo.score(X_test, y_test)
print'THIS ALGORITHM PREDICTS THE OUTPUT WITH A ACCURACY OF {}%'.format(accuracy*100)
