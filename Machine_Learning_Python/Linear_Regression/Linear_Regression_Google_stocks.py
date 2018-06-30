################################### LINEAR REGRESSION ###########################################
## GUILHERME CARVALHO PEREIRA
## IMPORTING LIBARIES 
import math, quandl
import sklearn 
import sklearn.linear_model 
import pandas as pd
import numpy as np
## GETING THE DATE SET FORM QUANDL
df_raw = quandl.get('WIKI/GOOGL')
## PREVIEWING THE DATA
#print(df_raw.head())
## GETING THE FEATURES MORE MEANINGFUL
df_n = df_raw[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#################################################################################################
## DEFINING THE SPECIAL RELATIONSHIP BETWEEN THE FEATURES
## HIGH_LOW_PERCENT (HL_PCT) => THIS FEATURE GIVE US THE PERCENT VOLATILITY OF THE STOCKS
df['HL_PCT'] = (df_n['Adj. High'] - df_n['Adj. Close'])/df_n['Adj. Close'] *100 
## PERCENT_CHANGE (PCT_CHANGE) => THIS FEATURE GIVE US THE DAILY MOVE OF THE STOCKS
df['PCT_CHANGE'] = (df_n['Adj. Close'] - df_n['Adj. Open'])/df_n['Adj. Open'] *100
## DEFINING THE DATAFRAME WITH THE SPECIAL RELATIONSHIPS
df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]
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
## PREVIEWING THE DATA
#print(df.head())
## DEFINING X AND y FOR OUR HYPOTHESIS
X = np.array(df.drop(['Label'], 1))
y = np.array(df['Label'])
## SCALING OUR FEATURES
X = sklearn.preprocessing.scale(X)
## DEVIDING OUR DATA INTO TRAINING AND DATA SET...TESTE SIZE IN %
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_slipt(X, y, teste_size = 0.2)
## DEFINING OUR LINEAR REGRESSION HYPOTHESIS
hypo = sklearn.linear_model.LinearRegression()
hypo.fit(X_train, y_train)
accuracy = hypo.score(X_test, y_test)
print(accuracy)
