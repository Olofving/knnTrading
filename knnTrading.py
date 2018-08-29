#Libraries
#Data manipulation
import numpy as np
import pandas as pd

#Plotting
import matplotlib.pyplot as plt

#ML Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Data fetching
from pandas_datareader import data as pandadr
#import fix_yahoo_finance as yf


##Script

#Read the data from Yahoo Finance
df = pandadr.get_data_yahoo('SPY', '2010-01-01', '2013-01-01')

df = df.dropna() #Droping NA-values
df = df[['Open', 'High', 'Low', 'Close']]

#Predictor variabales
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low
df = df.dropna()

X = df[['Open-Close', 'High-Low']] #Predictor variable

#Target varible
#Target variable Y is defined as; if SP500 closed above its
#previous day we will store +1 as buy signal, and if it
#closed below we will store -1 as sell signal
Y = np.where(df['Close'].shift(-1)>df['Close'],1,-1)

#Splitting the data set with 70-30, train-test ratio
split = int(0.7*len(df))

Xtrain = X[:split]
Ytrain = Y[:split]

Xtest = X[split:]
Ytest = Y[split:]

#Instantiate KNN algorithm 
k = 15
knn = KNeighborsClassifier(n_neighbors=k)

#Fit the model
knn.fit(Xtrain,Ytrain)

#Accuracy Score
accuracytrain = accuracy_score(Ytrain, knn.predict(Xtrain))
accuracytest = accuracy_score(Ytest,knn.predict(Xtest))

##Trading strategy

#Predicted signal
df['Predicted_Signal'] = knn.predict(X)

#SPY cumulative returns
df['SPY_Returns'] = np.log(df['Close']/df['Close'].shift(1))
Cumulative_SPY_Returns = df[split:]['SPY_Returns'].cumsum()*100

#Cumulative Strategy Returns
df['Strategy_Returns'] = df['SPY_Returns']*df['Predicted_Signal'].shift(1)
Cumulative_Strategy_Returns = df[split:]['Strategy_Returns'].cumsum()*100

##Plots
plt.figure(figsize=(20,10))
plt.plot(Cumulative_SPY_Returns, color = 'r', label = 'SPY Returns')
plt.plot(Cumulative_Strategy_Returns, color = 'b', label = 'Strategy Returns')
plt.title(str(k))
plt.legend()
plt.show()



  