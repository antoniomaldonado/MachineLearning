from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd


def train_with(training_data, training_label, epoc):
    # Initialising the ANN
    classifier = Sequential()
    classifier.add(Dense(units=16, activation='relu', input_dim=17))
    classifier.add(Dense(units=8, activation='relu'))
    classifier.add(Dense(units=6, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy')
    classifier.fit(training_data, training_label, batch_size=1, epochs=epoc)
    return classifier


def extract_and_prepare_data():
    global training_data, training_label, test_data, test_label
    data_frame = pd.read_csv('data/primary-tumor-lung-breast.data')
    data_frame.replace('?', -99999, inplace=True)
    all_data = np.array(data_frame.drop(['class'], 1))
    all_label = np.array(data_frame['class'])
    training_data = all_data[:int(round(len(all_data) * 0.8))]
    training_label = all_label[:int(round(len(all_data) * 0.8))]
    training_label = training_label.reshape(len(training_label), -1)
    test_data = all_data[int(round(len(all_data) * 0.8)):]
    test_label = all_label[:int(round(len(all_data) * 0.8))]
    test_label = test_label.reshape(len(test_label), -1)
    return training_data, training_label, test_data, test_label


if __name__ =='__main__':
    # Data prep
    training_data, training_label, test_data, test_label = extract_and_prepare_data()

    # Train
    classifier = train_with(training_data, training_label, 100)

    # Predict
    Y_pred = classifier.predict(test_data)
    Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred ]

    total = 0
    correct = 0
    wrong = 0
    for i in Y_pred:
        total=total+1
        if(test_label[i,0] == Y_pred[i]):
            correct+=1
        else:
            wrong+=1

    print("Total " + str(total))
    print("Correct " + str(correct))
    print("Wrong " + str(wrong))
