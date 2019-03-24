import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import style
import Utils
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression


def get_forecast_size(df, num):
    return int(math.ceil(num*len(df)))


def extract_and_prepare_data(quandl_code, feat1, feat2, num):
    # read from file
    df = Utils.data_frame_ini(quandl_code, feat1, feat2,)

    forecast_size = get_forecast_size(df, num)
    # add label field to df
    df['label'] = df[feat1].shift(-forecast_size)

    # Features
    features = np.array(df.drop(['label'],1))
    # standarize the dataset
    features = preprocessing.scale(features)

    # features we will use to predict
    recent_features = features[-forecast_size:]
    # features we will use to train
    features = features[:-forecast_size]

    df.dropna(inplace=True)

    return df, features, recent_features


def extract_and_train(feat1, feat2, quandl_code, split_data_percent):
    # Features
    df, features, recent_features = extract_and_prepare_data(
        quandl_code, feat1, feat2, split_data_percent)
    # Label
    label = np.array(df['label'])
    # Data prep
    training_data, test_data, training_label, test_label = \
        model_selection.train_test_split(
            features, label, test_size=split_data_percent)
    # train
    classifier = train_with(training_data, training_label)
    return classifier, df, recent_features, test_data, test_label


def train_with(training_data, training_label):
    # use n_jobs=-1 for as many jobs as possible
    classifier = LinearRegression(n_jobs=10)

    # different regression classification algorithm. 'SVR'
    # clf = svm.SVR(kernel='poly', gamma='auto')

    # training
    classifier.fit(training_data, training_label)
    return classifier


def plot_graph(df, feat1, forecast_set, field_name):
    style.use('ggplot')
    Utils.append_forecast_in_dataframe(df, forecast_set, 84600, field_name)
    df[feat1].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def save_training(classifier, df, recent_features, test_data, test_label):
    Utils.pickle_dump(classifier, 'classifier')
    Utils.pickle_dump(df, 'df')
    Utils.pickle_dump(recent_features, 'recent_features')
    Utils.pickle_dump(test_data, 'test_data')
    Utils.pickle_dump(test_label, 'test_label')


def load_training(classifier, df, recent_features, test_data, test_label):
    return Utils.pickle_load('classifier'),\
           Utils.pickle_load('df'),\
           Utils.pickle_load('recent_features'),\
           Utils.pickle_load('test_data'),\
           Utils.pickle_load('test_label')


if __name__ =='__main__':
    # split percentage of training data and prediction data
    split_data_percent = 0.01

    # Palladium prices features from https://www.quandl.com
    quandl_code = 'LPPM/PALL'
    feat1 = 'USD AM'
    feat2 = 'USD PM'

    # Avoid training if computations were already saved to disk
    classifier, df, recent_features, test_data, test_label = load_training(
        'classifier', 'df', 'recent_features', 'test_data', 'test_label')

    if classifier is None:
        # Extract and train the classifier
        classifier, df, recent_features, test_data, test_label = extract_and_train(
            feat1, feat2, quandl_code, split_data_percent)

    # Go predict
    forecast_set = classifier.predict(recent_features)
    accuracy = classifier.score(test_data, test_label)
    print('Size of our forecast: ' + str(get_forecast_size(df, split_data_percent)))
    print('Accuracy of our prediction: ' + str(accuracy))
    print('Forecast values: ' + str(forecast_set))

    # Save Training etc...
    save_training(classifier, df, recent_features, test_data, test_label)

    # Show graph
    plot_graph(df, feat1, forecast_set, 'Forecast')
