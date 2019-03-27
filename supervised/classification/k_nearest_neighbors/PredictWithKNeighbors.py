import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors


def extract_and_prepare_data():
    data_frame = pd.read_csv('data/primary-tumor.data')
    data_frame.replace('?',-99999, inplace=True)
    features = np.array(data_frame.drop(['class'], 1))
    label = np.array(data_frame['class'])
    return model_selection.train_test_split(features, label, test_size=0.2)


def train_with(training_data, label_data):
    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(training_data, label_data)
    return classifier


def predict(classifier, examples):
    print('Prediction : ' + str(classifier.predict(examples)))


if __name__ =='__main__':
    # Data prep
    training_data, test_data, training_label, test_label = extract_and_prepare_data()

    # Training
    classifier = train_with(training_data, training_label)
    print('K Neighbors classifier accuracy : ' + str(classifier.score(test_data, test_label)))

    # Data that we need to classify
    new_data = np.array([[2, 1, 1, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2],
                         [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2]])
    new_data = new_data.reshape(len(new_data), -1)

    # Expected classes [1, 22] 1-lung, 22-breast
    predict(classifier, new_data)
