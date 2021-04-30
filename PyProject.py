import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader as web
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM
from keras import Sequential
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

# Display options
pd.set_option("display.max_rows", None, "display.max_columns", None)

#computes indicators
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

            u = (1 / (n + 1)) * sum(h)
            d = (1 / (n + 1)) * sum(b)

            rsi.append(100 - (100 / (1 + (u / d))))

        return rsi


    # Stochastic Oscillator
    def oscill(data, n, d):
        K = []
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


    # Moving Average Convergence Divergence
    def MACD(data, n_large, n_small):
        list_small = []
        list_large = []
        ma_small = []
        ma_large = []

        for p in range(n_small, len(data)):
            for i in range(p - n_small, p):
                list_small.append(data['Adj Close'][i])

            small = (1 / (n_small + 1)) * sum(list_small)
            ma_small.append(small)

        for p in range(n_large, len(data)):
            for i in range(p - n_large, p):
                list_large.append(data['Adj Close'][i])

            large = (1 / (n_small + 1)) * sum(list_large)
            ma_large.append(large)

        return ma_small, ma_large;


    # Average Directional Index
    def ADX(data, n):

        true_range = [0]

        for p in range(1, len(data)):
            rnge = []

            high_low = abs(data['High'][p] - data['Low'][p])
            rnge.append(high_low)
            high_close = abs(data['High'][p] - data['Adj Close'][p - 1])
            rnge.append(high_close)
            low_close = abs(data['Low'][p] - data['Adj Close'][p - 1])
            rnge.append(low_close)

            true_range.append(max(rnge))

            if true_range[p] == 0:
                true_range[p] = true_range[p - 1]

        DM_plus = []
        DM_minus = []

        for p in range(0, len(data)):
            if (data['High'][p] - data['High'][p - 1]) >  (data['Low'][p - 1] - data['Low'][p]):
                DM_plus.append((data['High'][p] - data['High'][p - 1]))
                DM_minus.append(0)
            else:
                DM_minus.append((data['Low'][p - 1] - data['Low'][p]))
                DM_plus.append(0)

        rnge = 0
        minus = 0
        plus = 0

        for p in range(0, n):
            rnge = rnge + true_range[p]
            minus = minus + DM_minus[p]
            plus = plus + DM_plus[p]

        smooth_range = []
        smooth_plus = []
        smooth_minus = []

        for p in range(0, n):
            smooth_range.append(0)
            smooth_plus.append(0)
            smooth_minus.append(0)

        smooth_range.append(rnge)
        smooth_plus.append(plus)
        smooth_minus.append(minus)

        for p in range(n + 1, len(data)):
            avg_range = smooth_range[p - 1] / n
            avg_minus = smooth_minus[p - 1] / n
            avg_plus = smooth_plus[p - 1]/ n

            smooth_range.append(smooth_range[p - 1] - avg_range + true_range[p])
            smooth_minus.append(smooth_minus[p - 1] - avg_minus + DM_minus[p])
            smooth_plus.append(smooth_plus[p - 1] - avg_plus + DM_plus[p])

        indicator_plus = [0]
        indicator_minus = [0]

        for p in range(1, len(data)):
                indicator_plus.append((smooth_plus[p] / true_range[p]) * 100)
                indicator_minus.append((smooth_minus[p] / true_range[p]) * 100)

        dx = [0, 0, 0, 0, 0]

        for p in range(n, len(data)):
            dx.append((abs((indicator_plus[p] - indicator_minus[p]) / (indicator_plus[p] + indicator_minus[p]))) * 100)

        adx = []

        for p in range(0, len(data)):
            adx.append((1 / n) * sum(dx[p - n : p]))

        return adx;


    # On-Balance Volume
    def OBV(data):
        obv = [0]

        for p in range(1, len(data)):
            if data['Adj Close'][p] > data['Adj Close'][p - 1]:
                obv.append(obv[p - 1] + data['Volume'][p])
            elif data['Adj Close'][p] < data['Adj Close'][p - 1]:
                obv.append(obv[p - 1] - data['Volume'][p])
            else:
                obv.append(obv[p - 1])

        return obv;


    rsi = 9
    so = 14
    ma_so = 5
    ma = 20
    sd_boll = 2
    macd_small = 12
    macd_large = 26
    adx_length = 14

    RSI = RSI(data, rsi) # 9-days RSI
    K, D = oscill(data, so, ma_so) # 14-days SO & 5-days moving average
    MA, boll_up, boll_dw = boll(data, sd_boll, ma) # 20-days MA and 2-sd bollinger bands
    macd_short, macd_long = MACD(data, macd_large, macd_small) # 12 & 26 days moving averages
    adx = ADX(data, adx_length) # 14 days ADX & positive and negative indicators
    obv = OBV(data) # On Balance Volume

    # removing NAs
    RSI = RSI[(macd_large - rsi) : len(RSI)]
    K = K[(macd_large - so) : len(K)]
    D = D[(macd_large - (so + ma_so)) : len(D)]
    MA = MA[(macd_large - ma) : len(MA)]
    boll_up = boll_up[(macd_large - ma): len(boll_up)]
    boll_dw = boll_dw[(macd_large - ma): len(boll_dw)]
    macd_short = macd_short[(macd_large - macd_small) : len(macd_short)]
    adx = adx[macd_large : len(adx)]
    obv = obv[macd_large : len(obv)]

    df = {'RSI': RSI, 'D': D, 'MA': MA, 'boll_up': boll_up,
          'boll_dw': boll_dw, 'MACD_short' : macd_short, 'MACD_long' : macd_long,
          'adx' : adx, 'OBV' : obv}
    X = pd.DataFrame(df) # coercing indicators into dataframe

    return X
# encodes data into buy/hold/sell and add indicators
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

    X = indicators(data)  # computes indicators

    data.drop(['High', 'Low', 'Open', 'Close', 'Volume'], axis = 1, inplace = True)

    data.drop(data.index[len(data) - 1], axis=0, inplace=True) #remove last observation where position is NA

    data.insert(1, 'position', list) # add position

    data.drop(data.index[0: (len(data) - len(X))], axis=0, inplace=True)  # remove first observations where indicators are NA
    X.reset_index(drop=True, inplace=True)
    data.reset_index(drop=True, inplace=True)  # reseting index in data and X before concat
    data = pd.concat([data, X], axis=1)  # add X

    return data;
# transform the indicators from raw value to signal
def transform(data):
    RSI_signal = [4]
    D_signal = [4]
    boll_signal = [2]
    MACD_signal = [2]

    for p in range(1, len(data)):
        if data['RSI'][p] > 70 and data['RSI'][p - 1] < 70:
            RSI_signal.append(0)
        elif data['RSI'][p] < 70 and data['RSI'][p - 1] > 70:
            RSI_signal.append(1)
        elif data['RSI'][p] > 30 and data['RSI'][p - 1] < 30:
            RSI_signal.append(2)
        elif data['RSI'][p] < 30 and data['RSI'][p - 1] > 30:
            RSI_signal.append(3)
        else:
            RSI_signal.append(4)

        if data['D'][p] > 80 and data['D'][p - 1] < 80:
            D_signal.append(0)
        elif data['D'][p] < 80 and data['D'][p - 1] > 80:
            D_signal.append(1)
        elif data['D'][p] > 20 and data['D'][p - 1] < 20:
            D_signal.append(2)
        elif data['D'][p] < 20 and data['D'][p - 1] > 20:
            D_signal.append(3)
        else:
            D_signal.append(4)

        if data['Adj Close'][p] > data['boll_up'][p]:
            boll_signal.append(0)
        elif data['Adj Close'][p] < data['boll_dw'][p]:
            boll_signal.append(1)
        else:
            boll_signal.append(2)

        if data['MACD_short'][p] > data['MACD_long'][p] and data['MACD_short'][p - 1] < data['MACD_long'][p - 1]:
            MACD_signal.append(0)
        elif data['MACD_short'][p] < data['MACD_long'][p] and data['MACD_short'][p - 1] > data['MACD_long'][p - 1]:
            MACD_signal.append(1)
        else:
            MACD_signal.append(2)

    for p in range(0, len(data)):
        if data['position'][p] == 2:
            data['position'][p] = 1
        elif data['position'][p] == 3:
            data['position'][p] = 0

    data['RSI'] = RSI_signal
    data['D'] = D_signal
    data['boll'] = boll_signal
    data['MACD_long'] = MACD_signal

    data.drop(['MACD_short', 'MA', 'boll_up', 'boll_dw'], axis = 1, inplace = True)

    return data;
# Splits data into training and testing
def test_train_split(data, train):

    slice = train * len(data)
    slice = int(slice) # slice = index where test set begins

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
def test_train_SP(data, train):
    data = data[data_SP.index[26]: data.index[len(data) - 1]]

    slice = train * len(data)
    slice = int(slice)

    data_copy = data[data.index[0]: data.index[slice - 1]]
    data_test = data[data.index[slice]: data.index[len(data) - 1]]  # slicing data
    data = data_copy

    return data, data_test
# plots of conditional distributions with respect to inputs
def inputPlots(data):
    fig, axs = plt.subplots(2, 3)

    axs[0, 0].scatter(y = data['RSI'], x = data['position'])
    axs[0, 0].set_title('Relative Strenght Index')
    axs[0, 0].set_xlabel('position')
    axs[0, 0].set_ylabel('RSI')

    axs[0, 1].scatter(y=data['MA'], x=data['position'])
    axs[0, 1].set_title('20 days moving average')
    axs[0, 1].set_xlabel('position')
    axs[0, 1].set_ylabel('moving average')

    axs[0, 2].scatter(y = data['D'], x = data['position'])
    axs[0, 2].set_title('smoothed stochastic oscillator')
    axs[0, 2].set_xlabel('position')
    axs[0, 2].set_ylabel('oscillator')

    axs[1, 0].scatter(y = data['adx'], x = data['position'])
    axs[1, 0].set_title('Average Directional Index')
    axs[1, 0].set_xlabel('position')
    axs[1, 0].set_ylabel('ADX')

    axs[1, 1].scatter(y = data['OBV'], x = data['position'])
    axs[1, 1].set_title('On-Balance Volume')
    axs[1, 1].set_xlabel('position')
    axs[1, 1].set_ylabel('OBV')

    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.show()
# Neural net architecture
def NeuralNet():
        NN = Sequential()

        NN.add(layers.Dense(10, activation='relu'))
        NN.add(layers.Dense(10, activation='relu'))
        NN.add(layers.Dense(2, activation='softmax'))

        NN.compile(optimizer='adam',
                   loss=keras.losses.SparseCategoricalCrossentropy(),
                   metrics=keras.metrics.SparseCategoricalCrossentropy(),
                   )

        return NN
# AUC + ROC + confusion matrix (in case of binary buy/sell classification)
def results(y, pred_class, model):
    fig, axs  = plt.subplots(1, 2, figsize = (10, 5))

    fpr, tpr, thr = roc_curve(y, pred_class)
    roc_auc = auc(fpr, tpr)
    print(model, roc_auc)


    axs[0].plot(fpr, tpr, lw=2, alpha=0.7, label=model)
    axs[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')

    axs[1].plot(history.history['loss'])
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')

    conf_mat = confusion_matrix(pred_class, y)
    report = classification_report(pred_class, y)
    print(report, conf_mat)
# Second Neural Net architecture
def ImprovedNeuralNet():
        NN = Sequential()

        NN.add(layers.Dense(32, activation='relu'))
        NN.add(layers.Dense(32, activation='relu'))
        NN.add(layers.Dense(32, activation='relu'))
        NN.add(layers.Dense(1, activation='sigmoid'))

        NN.compile(optimizer='adam',
                   loss=keras.losses.BinaryCrossentropy(),
                   metrics=keras.metrics.BinaryCrossentropy(),
                   )

        return NN
# Encodes predictions into string variables easier to read (deprecated)
def output_encode(pred_class, data):
    list = []

    for p in range(0, len(pred_class)):
        if pred_class[p] == 1:
            list.append('buy')
        elif pred_class[p] == 0:
            list.append('sell')


    data.insert(3, 'pred_string', list)
    data.insert(4, 'pred_pos', pred_class)

    return data
# Computes profits based on predictions and returns for the S&P500 same period
def profits(data, amount):
    init = amount
    profits = []
    total = [amount]
    returns = []
    realized_returns = []

    for p in range(0, len(data) - 1):

        tx = (data['Adj Close'][p + 1] - data['Adj Close'][p]) / data['Adj Close'][p]

        if data['position'][p] == 0 and data['pred'][p] == 0:
            profit = - (init * tx)
            init = init + profit
            if init > amount:
                realized_returns.append(init - amount)
                init = amount
            else:
                realized_returns.append(0)
        elif data['position'][p] == 0 and data['pred'][p] == 1:
            profit = init * tx
            init = init + profit
            realized_returns.append(0)
        elif data['position'][p] == 1 and data['pred'][p] == 1:
            profit = init * tx
            init = init + profit
            if init > amount:
                realized_returns.append(init - amount)
                init = amount
            else:
                realized_returns.append(0)
        elif data['position'][p] == 1 and data['pred'][p] == 0:
            profit = - (init * tx)
            init = init + profit
            realized_returns.append(0)
        else:
            print('error')

        profits.append(profit)

        total.append(init)

        returns.append(tx)

    profits.append(0)
    returns.append(0)
    realized_returns.append(0)

    data['profit'] = profits
    data['total'] = total
    data['%returns'] = returns
    data['realized_returns'] = realized_returns

    sum_returns = sum(data['%returns'])
    print(round(sum_returns, 2))
    sum_profits = sum(data['profit'])
    print(round(sum_profits, 2))
    sum_realized_returns = sum(data['realized_returns'])
    print(round(sum_realized_returns, 2))
    percentage_gain = (sum_realized_returns * 100) / amount
    print(round(percentage_gain, 2), '%')  # sum profits and compute % return

    return data;
def profits_SP(data, amount):
    init = amount
    profits = []
    total = [amount]
    returns = []
    realized_returns = []

    for p in range(0, len(data) - 1):

        tx = (data['Adj Close'][p + 1] - data['Adj Close'][p]) / data['Adj Close'][p]

        if tx > 0:
            profit = init * tx
            init = init + profit
            if init > amount:
                realized_returns.append(init - amount)
                init = amount
            else:
                realized_returns.append(0)
        else:
            profit = init * tx
            init = init + profit
            realized_returns.append(0)

        profits.append(profit)

        total.append(init)

        returns.append(tx)

    profits.append(0)
    returns.append(0)
    realized_returns.append(0)

    data['profit'] = profits
    data['total'] = total
    data['%returns'] = returns
    data['realized_returns'] = realized_returns

    sum_returns = sum(data['%returns'])
    sum_profits = sum(data['profit'])
    sum_realized_returns = sum(data['realized_returns'])
    percentage_gain = (sum_realized_returns * 100) / amount

    print('sum profits : ', round(sum_profits, 2))
    print('sum returns : ', round(sum_returns, 2))
    print('sum realized returns : ', round(sum_realized_returns, 2))
    print('total realized return : ', round(percentage_gain, 2), '%')  # sum profits and compute % return

    return data;
def profitsV2(data, amount):
    available = amount
    total = 0
    profit = 0
    realized_returns = []
    returns = []
    available_amount = []
    invested_amount = []
    profits = []

    for p in range(0, len(data) - 1):

        rate = (data['Adj Close'][p + 1] - data['Adj Close'][p]) / data['Adj Close'][p]
        returns.append(rate)

        if data['pred'][p] == 0:
            available = total
            profit = 0
            profits.append(profit)
            available_amount.append(available)
            invested_amount.append(0)
        else:
            if total == 0:
                total = available
            else:
                total = total

            available = 0
            profit = total * rate
            total = total + profit
            profits.append(profit)
            available_amount.append(available)
            invested_amount.append(total)

        if total > amount:
            realized_returns.append(total - amount)
            total = amount
        else:
            realized_returns.append(0)
            total = total

    profits.append(0)
    returns.append(0)
    realized_returns.append(0)
    invested_amount.append(0)
    available_amount.append(total)

    data['available'] = available_amount
    data['invested'] = invested_amount
    data['%returns'] = returns
    data['profit'] = profits
    data['realized_returns'] = realized_returns

    sum_returns = sum(data['%returns'])
    sum_profits = sum(data['profit'])
    sum_realized_returns = sum(data['realized_returns'])
    percentage_gain = (sum_realized_returns * 100) / amount

    print('sum profits : ', round(sum_profits, 2))
    print('sum returns : ', round(sum_returns, 2))
    print('sum realized returns : ', round(sum_realized_returns, 2))
    print('total realized return', round(percentage_gain, 2), '%')  # sum profits and compute % return

    return data;
# Counts the number of time the model is right/wrong
def accuracy(data):
    proportion = []

    for p in range(0, len(data) - 1):

        tx = (data['Adj Close'][p + 1] - data['Adj Close'][p]) / data['Adj Close'][p]

        if data['position'][p] == data['pred'][p]:
            proportion.append(1)
        else:
            proportion.append(0)

    proportion.append(0)

    data['proportion'] = proportion

    return data;

# Choosing assets
ticker = 'AAPL' #Walmart:WMT - Apple:AAPL - AirFrance:AF.PA - Tesla:TSLA
ticker_SP = '^GSPC' # ticker for the S&P500

start = dt.datetime(2010,8,1) # series starts on 2010/08/01
end = dt.datetime(2019,12,31) # ends on 2019/12/31

# importing data from yahoo API
data = web.DataReader(ticker, 'yahoo', start, end)
data_SP = web.DataReader(ticker_SP, 'yahoo', start, end)

# plot of the serie
plt.plot(data['Adj Close'])
plt.show()
plt.title(ticker)
plt.ylabel('Adj Close')
plt.xlabel('Time')

data = encode(data) # encodes data into buy/hold/sell and add indicators

signal_data = transform(data) # transform the indicators from raw value to signal

train_size = 0.7 # 70% of the data to training 30% to testing

# Spliting data into training and testing sets
data, data_test, X, y, X_test, y_test = test_train_split(data, train_size)
data_SP, data_SP_test = test_train_SP(data_SP, train_size)

# position frequencies (test VS train)
plt.bar(data['position'].value_counts().index, data['position'].value_counts().values)
plt.bar(data_test['position'].value_counts().index, data_test['position'].value_counts().values)
plt.title(ticker)
plt.show()

inputPlots(data) # plots of conditional distributions with respect to inputs

# Logit
logit = sm.MNLogit(y, X)
logit_fit = logit.fit(method = 'newton', maxiter = 100)
logit_fit.summary()

#Normalize features
scaler = MinMaxScaler(feature_range = [0, 1]).fit(X)
X = scaler.transform(X) # Train data

scaler = MinMaxScaler(feature_range = [0, 1]).fit(X_test)
X_test = scaler.transform(X_test) # Test data

# Standard Neural Net
NN = NeuralNet() # creates Neural Net
history = NN.fit(X, y, epochs = 500) # fits the model
pred = NN.predict(X) # Predicted probabilities on train data
pred_class = pred.argmax(axis = -1) # Predicted class on train data
pred_proba =  NN.predict_proba(X)[:, 1] # computes predicted probabilities for each class

results(y, pred_class, NN) # AUC + ROC + confusion matrix (in case of binary buy/sell classification)

data_test['pred'] = pred_class_test # adding predictions as 0/1 to the dataframe
data_test = accuracy(data_test) # Counts the number of time the model is right/wrong

# Improved Neural Net
ImprovNN = ImprovedNeuralNet() # creates Neural Net
history = ImprovNN.fit(X, y, epochs = 1000) # fits the model
pred = ImprovNN.predict(X) # Predicted probabilities on train data
pred_class = ImprovNN.predict_classes(X) # Predicted class on train data
pred_class_test = ImprovNN.predict_classes(X_test) # Predicted class on test data

results(y, pred_class, ImprovNN) # AUC + ROC + confusion matrix (in case of binary buy/sell classification)

data_test['pred'] = pred_class_test # adding predictions as 0/1 to the dataframe
data_test = accuracy(data_test) # Counts the number of time the model is right/wrong

# linear SVM
SVM = svm.SVC()
SVM_fit = SVM.fit(X, y)
pred_class = SVM.predict(X)
pred_class_test = SVM.predict(X_test)

results(y, pred_class, SVM) # AUC + ROC + confusion matrix (in case of binary buy/sell classification)

data_test['pred'] = pred_class_test # adding predictions as 0/1 to the dataframe
data_test = accuracy(data_test) # Counts the number of time the model is right/wrong

# RBF kernel SVM
grid = {
	'C': [0.001, 0.01, 0.1, 1, 10, 100],
	'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
	'kernel': ['rbf']} # grid of values to evaluate

rbf_SVM = svm.SVC(max_iter = 1000)
grid_search = GridSearchCV(rbf_SVM, param_grid = grid, refit = True)
rbf_SVM_fit = grid_search.fit(X, y)
pred_class_test = grid_search.predict(X_test)

print(grid_search.best_params_) # displays the best set of parameters

results(y, pred_class, rbf_SVM) # AUC + ROC + confusion matrix (in case of binary buy/sell classification)

data_test['pred'] = pred_class_test # adding predictions as 0/1 to the dataframe
data_test = accuracy(data_test) # Counts the number of time the model is right/wrong

# Profits
amount = 1000

data_test = profitsV2(data_test, amount) # computes profits made with specified initial investment
data_SP_test = profits_SP(data_SP_test, amount) # same for S&P

plt.plot(data_test['profit'])
plt.show
plt.plot(data_test['total'])
plt.show
plt.plot(data_test['realized_returns'])
plt.show()

#LSTM
def lstm():
    for p in range(0, len(data)):
        if data['position'][p] == 2:
            data['position'][p] = 1
        elif data['position'][p] == 3:
            data['position'][p] = 0

    scaler = MinMaxScaler(feature_range = [0, 1]).fit(X)
    X = scaler.transform(X)

    n = 14
    Xtrain = []
    ytrain = []

    for p in range(n, len(X)):
        Xtrain.append(X[p - n : p, : X.shape[1]])
        ytrain.append(y[p])

    Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))


    def lstm_net():
        model = Sequential()
        model.add(LSTM(64, input_shape = (Xtrain.shape[1], Xtrain.shape[2])))
        model.add(layers.Dense(1))

        model.compile(loss = keras.losses.BinaryCrossentropy(),
                    metrics=keras.metrics.BinaryCrossentropy(),
                    optimizer="adam")

        return model


    lstm = lstm_net()

    history = lstm.fit(Xtrain, ytrain, epochs = 100)

    pred = lstm.predict(Xtrain) # Predicted probabilities on train data
    pred_class = pred.argmax(axis = -1) # Predicted class on train data

    results(ytrain, pred_class, lstm)



