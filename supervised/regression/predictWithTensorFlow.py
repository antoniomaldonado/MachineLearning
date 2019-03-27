from __future__ import absolute_import, division, print_function
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


def extract_and_format():
    # Original data from https://archive.ics.uci.edu
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']
    dataset = pd.read_csv('data/auto-mpg.data', names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
    dataset = dataset.dropna()

    # convert to one-hot format
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    dataset.tail()

    return dataset


def build_model():

    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1) # no activation parameter makes it linear. We can use this for regression
    ])

    # RMS to mitigate Gradients propagation
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error', # loss function
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error']) # measure how well the model is doing
    return model


# Predict fuel efficiency
if __name__ =='__main__':

    dataset = extract_and_format()

    # 80% for training
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    # 20% to test
    test_dataset = dataset.drop(train_dataset.index)

    # Describe will generate the mean and standard deviation
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    train_stats

    # Normalize data
    train_labels = train_dataset.pop('MPG')
    normalized_train_data =  (train_dataset - train_stats['mean']) / train_stats['std']
    test_labels = test_dataset.pop('MPG')
    normalized_test_data =  (test_dataset - train_stats['mean']) / train_stats['std']

    # Build the model
    model = build_model()

    # Function to correct when the error start to grow
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Train the model
    model.fit(normalized_train_data, train_labels, epochs=2000, # epochs: How many times we are going to train
                        validation_split = 0.2, verbose=0, callbacks=[early_stop])

    # Evaluate accuracy of the model
    loss, mae, mse = model.evaluate(normalized_test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
    print("Testing set Mean Sqr Error: {:5.2f} MPG".format(mse))

