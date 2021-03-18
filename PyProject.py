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
from sklearn.metrics import confusion_matrix, classification_report

seed(1000) # ensures reproductible results

# importing data from yahoo API
ticker = 'AF.PA'

start = dt.datetime(2010,1,1) # series starts on 2010/01/01 ends on 2019/12/31
end = dt.datetime(2019,12,31)

data = web.DataReader(ticker, 'yahoo', start, end)

# plot
#plt.plot(data)
#plt.show()

# computing indicators

def indicators(data):

    # Relative Strength Index
    def RSI(data, n):
        rsi = []

        for p in range (n, len(data)):
            h = []
            b = []
            for i in range(0, n):
                diff = 100 * ((data['Adj Close'][(p - n + i) + 1] - data['Adj Close'][p - n + i]) / data['Adj Close'][p - n + i])

                if diff < 0:
                    b.append(abs(diff))
                elif diff > 0:
                    h.append(diff)

            u = (1 / n) * sum(h)
            d = (1 / n) * sum(b)

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

            for i in range(p - d, p):
                sum = sum + K[i]
                Kma = (1 / d) * sum

            ma.append(Kma)

        return K, ma;


    # Bollinger Bands
    def boll(data, k, n):
        MA = []
        boll_up = []
        boll_dw = []

        for p in range(n, len(data)):
            sum = 0
            var = 0

            for i in range(p - n, p):
                sum = sum + data['Adj Close'][i]
                ma = (1 / n) * sum

            for i in range(p - n, p):
                spread = (data['Adj Close'][i] - ma) ** 2
                var = var + spread

            sigma = np.sqrt((1 / n) * var)

            up = ma + k * sigma
            dw = ma - k * sigma

            MA.append(ma)
            boll_up.append(up)
            boll_dw.append(dw)

        return MA, boll_up, boll_dw;

    RSI = RSI(data, 9) # 9-days RSI
    K, D = oscill(data, 14, 5) # 14-days SO & 5-days moving average
    MA, boll_up, boll_dw = boll(data, 2, 20) # 20-days MA and 2-sd bollinger bands

    return RSI, K, D, MA, boll_up, boll_dw

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

def test_train_split(data, train):

    slice = train * len(data)
    slice = int(slice) # slice = index where test begins

    data_copy = data[data.index[0] : data.index[slice - 1]]
    data_test = data[data.index[slice] : data.index[len(data) - 1]]
    data = data_copy

    return data, data_test;


data, data_test = test_train_split(data, 0.7)


def encode(data):

    RSI, K, D, MA, boll_up, boll_dw = indicators(data)  # computes indicators

    list = []

    for p in range(1, len(data)):
        if data['Adj Close'][p] > data['Adj Close'][p - 1]:
            list.append(1)
        elif data['Adj Close'][p] == data['Adj Close'][p - 1]:
            list.append(2)
        elif data['Adj Close'][p] < data['Adj Close'][p - 1]:
            list.append(3)
        else:
            print('error')

    data.drop(['High', 'Low', 'Open', 'Close'], axis = 1, inplace = True)
    data.drop(data.index[0], axis = 0, inplace = True)

    data.insert(2, 'position', list)

    RSI = RSI[11: len(RSI)]
    D = D[2: len(D)]
    df = {'RSI': RSI, 'D': D, 'MA': MA, 'boll_up': boll_up, 'boll_dw': boll_dw}
    X = pd.DataFrame(df)

    y = data['position']
    y = y[19: len(y)]

    return data, X, y;


data, X, y = encode(data) # encodes data into buy/hold/sell and creates X and y np.arrays
data_test, X_test, y_test = encode(data_test) # encodes data into buy/hold/sell and creates X and y np.arrays

# Neural Net

def NeuralNet():

    NN = Sequential()

    NN.add(layers.Dense(3, activation = 'relu'))
    NN.add(layers.Dense(3, activation = 'relu'))
    NN.add(layers.Dense(3, activation = 'relu'))
    NN.add(layers.Dense(4, activation = 'softmax'))

    NN.compile(optimizer='adam',
               loss=keras.losses.SparseCategoricalCrossentropy(),
               metrics=keras.metrics.SparseCategoricalAccuracy(),
               )

    return NN


NN = NeuralNet() # creates Neural Net

NN.fit(X, y, epochs = 15) # fits the model

pred = NN.predict(X_test)
pred_class = pred.argmax(axis = -1) # Predicted class on test data

pred_proba =  NN.predict_proba(X_test)[:, 1] # computes predicted probabilities for each class


def output_encode(pred_class, data):

    list = []

    for p in range(0, len(pred_class) - 1):
        if pred_class[p] == 1:
            list.append('buy')
        elif pred_class[p] == 2:
            list.append('hold')
        else:
            list.append('sell')

    data.drop(data.index[0 : 20], axis = 0, inplace = True)

    data.insert(3, 'pred_string', list)
    data.insert(4, 'pred_pos', pred_class[0 : -1])

    return data


data_test = output_encode(pred_class, data_test) # encodes output as categorical variables


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

# fpr, tpr, thr = roc_curve(y_test, pred_class)
# roc_auc = auc(fpr, tpr)
# print(model, roc_auc)
# plt.plot(fpr, tpr, lw=2, alpha=0.7, label=model)
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

# Confusion matrix and classification report
conf_mat = confusion_matrix(y_test, pred_class)
report = classification_report(y_test, pred_class)
print(report, conf_mat)
