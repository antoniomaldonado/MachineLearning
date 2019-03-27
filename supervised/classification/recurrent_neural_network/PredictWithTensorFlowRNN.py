import json, numpy as np,pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import math
from sklearn.metrics import mean_squared_error

ACTIVATION = 'linear'
INPUT_DIM = 30

def extract_data():

    # Data since 18 August 2017
    with open('data/lastYearBitcoinPrices.json') as f:
        content = f.read()

    prices = json.loads(content)['data']['history']

    df = pd.DataFrame(prices)
    df['price'] = pd.to_numeric(df['price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms').dt.date

    return df.groupby('timestamp').mean()['price']


def format_data(data):
    matrix = []
    for index in range(len(data)-INPUT_DIM+1):
        matrix.append(data[index:index+INPUT_DIM])
    return matrix


def split(price_matrix, train_size=0.8):
    price_matrix = np.array(price_matrix)
    row = int(round(train_size * len(price_matrix)))
    train = price_matrix[:row, :]
    train_x, train_y = train[:row,:-1], train[:row,-1]
    test_x, test_y = price_matrix[row:,:-1], price_matrix[row:,-1]
    return np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1)), \
           train_y, \
           np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1)), \
           test_y


def build_model():
    model = Sequential()
    model.add(LSTM(units=INPUT_DIM, return_sequences=True, input_shape=(None, 1)))
    model.add(Dense(units=32, activation=ACTIVATION))
    model.add(LSTM(units=INPUT_DIM, return_sequences=False))
    model.add(Dense(units=1, activation=ACTIVATION))
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model


if __name__ =='__main__':
    # Prepare data
    data = format_data(extract_data())
    train_x, train_y, test_x, test_y = split(data)

    # Build the RNN model
    model = build_model()

    # Train
    model.fit(x=train_x,y=train_y,batch_size=2, epochs=100,validation_split=0.05)

    # Predict
    trainPredict = model.predict(train_x)
    testPredict = model.predict(test_x)

    trainScore = math.sqrt(mean_squared_error(train_y, trainPredict))
    testScore = math.sqrt(mean_squared_error(test_y, testPredict))

    # Check the accuracy of our model
    print('Train Score: %.5f RMSE' % (trainScore))
    print('Test Score: %.5f RMSE' % (testScore))
