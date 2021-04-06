seed(1000) # ensures reproductible results

import pandas as pd
import numpy as np
from numpy.random import seed
import sklearn as sk
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader as web
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from keras import Sequential
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# importing data from yahoo API
ticker = 'TSLA'

start = dt.datetime(2010,1,1) # series starts on 2010/01/01 ends on 2019/12/31
end = dt.datetime(2019,12,31)

data = web.DataReader(ticker, 'yahoo', start, end)

#plot
plt.plot(data['Adj Close'])
plt.show()
plt.title(ticker)
plt.ylabel('Adj Close')
plt.xlabel('Time')

# computing indicators
def indicators(data):

    # Relative Strength Index
    def RSI(data, n):
        rsi = []
        n = n - 1

        for p in range (n, len(data)):
            h = []
            b = []

            for i in range(0, n):
                diff = 100 * ((data['Adj Close'][(p - n + i) + 1] - data['Adj Close'][p - n + i]) / data['Adj Close'][p - n + i])

                if diff < 0:
                    b.append(abs(diff))
                elif diff > 0:
                    h.append(diff)

            u = (1 / (n + 1)) * sum(h)
            d = (1 / (n + 1)) * sum(b)

            rsi.append(100 - (100 / (1 + (u / d))))

        return rsi


    # Stochastic Oscillator
    def oscill(data, n, d):
        K = []
        n = n - 1
        ma = []

        for p in range(n, len(data)):
            values = []
            close = data['Adj Close'][p]

            for i in range(p - n, p + 1):
                values.append(data['Adj Close'][i])

            high = max(values)
            low = min(values)

            K.append(((close - low) / (high - low)) * 100)

        for p in range(d, len(K)):
            sum = 0

            for i in range(p - d, p + 1):
                sum = sum + K[i]
                Kma = (1 / d) * sum

            ma.append(Kma)

        return K, ma;


    # Bollinger Bands
    def boll(data, k, n):
        n = n - 1
        MA = []
        boll_up = []
        boll_dw = []

        for p in range(n, len(data)):
            sum = 0
            var = 0

            for i in range(p - n, p + 1):
                sum = sum + data['Adj Close'][i]

            ma = (1 / (n + 1)) * sum

            for i in range(p - n, p + 1):
                spread = (data['Adj Close'][i] - ma) ** 2
                var = var + spread

            sigma = np.sqrt((1 / (n + 1)) * var)

            up = ma + (k * sigma)
            dw = ma - (k * sigma)

            MA.append(ma)
            boll_up.append(up)
            boll_dw.append(dw)

        return MA, boll_up, boll_dw;

    rsi = 9
    so = 14
    ma_so = 5
    ma = 20
    sd_boll = 2

    RSI = RSI(data, rsi) # 9-days RSI
    K, D = oscill(data, so, ma_so) # 14-days SO & 5-days moving average
    MA, boll_up, boll_dw = boll(data, sd_boll, ma) # 20-days MA and 2-sd bollinger bands

    # removing NAs
    RSI = RSI[(ma - rsi) : len(RSI)]
    K = K[(ma - so) : len(K)]
    D = D[(ma - (so + ma_so)) : len(D)]
    df = {'RSI': RSI, 'D': D, 'MA': MA, 'boll_up': boll_up, 'boll_dw': boll_dw}
    X = pd.DataFrame(df) # coercing indicators into dataframe

    return X

#fig, axs = plt.subplots(2)
#axs[0].plot(data['Adj Close'][0:100], 'red')
#axs[1].plot(RSI[0:100], color = 'steelblue')

#fig, axs = plt.subplots(2)
#axs[0].plot(data['Adj Close'][0:200], color = 'red')
#axs[1].plot(D, color = 'steelblue')

#index = range(0, len(MA)) # creates index (to be removed)

#plt.plot(index, MA, linestyle = '--', color = 'darkorchid')
#plt.plot(index, boll_up, linestyle = '-', color = 'forestgreen')
#plt.fill_between(index, MA, boll_up, facecolor = "forestgreen", alpha = 0.3)
#plt.fill_between(index, MA, boll_dw, facecolor = "firebrick", alpha = 0.3)
#plt.plot(index, boll_dw, linestyle = '-', color = 'firebrick')

def encode(data):

    list = []

    for p in range(0, len(data) - 1):
        if data['Adj Close'][p + 1] > data['Adj Close'][p]:
            list.append(1)
        elif data['Adj Close'][p + 1] == data['Adj Close'][p]:
            list.append(2)
        elif data['Adj Close'][p + 1] < data['Adj Close'][p]:
            list.append(3)
        else:
            print('error')

    data.drop(['High', 'Low', 'Open', 'Close', 'Volume'], axis = 1, inplace = True)

    X = indicators(data)  # computes indicators

    data.drop(data.index[len(data) - 1], axis=0, inplace=True) #remove last observation where position is NA
    data.insert(1, 'position', list) # add position
    data.drop(data.index[0 : (len(data) - len(X))], axis=0, inplace=True) # remove first observations where indicators are NA
    #data = data.join(X)
    X.reset_index(drop = True, inplace = True)
    data.reset_index(drop = True, inplace = True)
    data = pd.concat([data, X], axis = 1) # add X

    return data, X;


data, X = encode(data) # encodes data into buy/hold/sell and creates X array of predictors

# frequencies
plt.bar(data['position'].value_counts().index, data['position'].value_counts().values)
plt.title(ticker)
plt.show()

def test_train_split(data, train):

    slice = train * len(data)
    slice = int(slice) # slice = index where test begins

    data_copy = data[data.index[0] : data.index[slice - 1]]
    data_test = data[data.index[slice] : data.index[len(data) - 1]] # slicing data
    data = data_copy

    data.reset_index(drop = True, inplace = True) # reset index necessary for profits function
    data_test.reset_index(drop = True, inplace = True)

    y = data['position'] # re-creating X and y
    y.reset_index(drop = True, inplace =True)
    X = data.drop(['Adj Close', 'position'], axis = 1)
    X.reset_index(drop = True, inplace = True)

    y_test = data_test['position'] # creating X_test and y_test
    y_test.reset_index(drop=True, inplace=True)
    X_test = data_test.drop(['Adj Close', 'position'], axis = 1)
    X_test.reset_index(drop=True, inplace=True)

    return data, data_test, X, y, X_test, y_test;


data, data_test, X, y, X_test, y_test = test_train_split(data, 0.7)

# frequencies (test VS train)
plt.bar(data['position'].value_counts().index, data['position'].value_counts().values)
plt.bar(data_test['position'].value_counts().index, data_test['position'].value_counts().values)
plt.title(ticker)
plt.show()

#inputs plot
plt.scatter(y = data['RSI'], x = data['position'])
plt.scatter(y = data['MA'], x = data['position'])
plt.scatter(y = data['D'], x = data['position'])
plt.scatter(y = data['boll_up'], x = data['position'])
plt.scatter(y = data['boll_dw'], x = data['position'])

# Neural Net
def NeuralNet():

    NN = Sequential()

    NN.add(layers.Dense(3, activation = 'relu'))
    NN.add(layers.Dense(3, activation = 'relu'))
    NN.add(layers.Dense(3, activation = 'relu'))
    NN.add(layers.Dense(4, activation = 'softmax'))

    NN.compile(optimizer='adam',
               loss=keras.losses.SparseCategoricalCrossentropy(),
               metrics=keras.metrics.SparseCategoricalCrossentropy(),
               )

    return NN


NN = NeuralNet() # creates Neural Net

NN.fit(X, y, epochs = 500) # fits the model

pred = NN.predict(X_test)
pred_class = pred.argmax(axis = -1) # Predicted class on test data

pred_proba =  NN.predict_proba(X_test)[:, 1] # computes predicted probabilities for each class


def output_encode(pred_class, data):
    list = []

    for p in range(0, len(pred_class)):
        if pred_class[p] == 1:
            list.append('buy')
        elif pred_class[p] == 2:
            list.append('hold')
        else:
            list.append('sell')

    data.insert(3, 'pred_string', list)
    data.insert(4, 'pred_pos', pred_class)

    return data


data_test = output_encode(pred_class, data_test) # encodes output as categorical string variables


def profits(data, amount):
    init = amount
    profits = []

    for p in range(1, len(data)):

        tx = ((data['Adj Close'][p] - data['Adj Close'][p - 1]) / data['Adj Close'][p - 1])

        if data['position'][p - 1] == 1 and data['pred_pos'][p - 1] == 1:
            profit = amount * tx
        elif data['position'][p - 1] == 1 and data['pred_pos'][p - 1] == 2:
            profit = 0
        elif data['position'][p - 1] == 1 and data['pred_pos'][p - 1] == 3:
            profit = - (amount * tx)
        elif data['position'][p - 1] == 2 and data['pred_pos'][p - 1] == 1:
            profit = 0
        elif data['position'][p - 1] == 2 and data['pred_pos'][p - 1] == 2:
            profit = 0
        elif data['position'][p - 1] == 2 and data['pred_pos'][p - 1] == 3:
            profit = 0
        elif data['position'][p - 1] == 3 and data['pred_pos'][p - 1] == 1:
            profit = amount * tx
        elif data['position'][p - 1] == 3 and data['pred_pos'][p - 1] == 2:
            profit = 0
        elif data['position'][p - 1] == 3 and data['pred_pos'][p - 1] == 3:
            profit = - (amount * tx)
        else:
            print('error')

        profits.append(profit)

    profits.insert(0, 0)

    return profits


amount = 1000

profits = profits(data_test, amount) # computes profits made with specified initial investment

data_test.insert(3, 'profit', profits)

sum_profits = sum(data_test['profit'])
print(round(sum_profits, 2))
percentage_gain = sum_profits * 100 / amount
cumsum = np.cumsum(profits)
print(round(percentage_gain, 2),'%') # sum profits and compute % return

# plt.plot(data_test['profit'])
# plt.show
# plt.plot(cumsum)
# plt.show

# AUC + ROC (in case of binary buy/sell classification)

#fpr, tpr, thr = roc_curve(y_test, pred_class)
#roc_auc = auc(fpr, tpr)
#print(model, roc_auc)
#plt.plot(fpr, tpr, lw=2, alpha=0.7, label=model)
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.legend()
#plt.show()

# Confusion matrix and classification report
conf_mat = confusion_matrix(y_test, pred_class)
report = classification_report(y_test, pred_class)
print(report, conf_mat)
