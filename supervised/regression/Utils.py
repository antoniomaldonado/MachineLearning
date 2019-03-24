import datetime
import pickle
import numpy as np
import os
import quandl


def data_frame_ini(quandl_code, arg1, arg2):
    df = quandl.get(quandl_code)
    df = df[[arg1, arg2]]
    df.fillna(-99999, inplace=True)
    return df


def pickle_dump(classifier, file_name):
    # dump serialized objects to file
    with open('data/'+file_name+'.picle', 'wb') as f:
        pickle.dump(classifier, f)


def pickle_load(file_name):
    try:
        file = 'data/'+file_name+'.picle'
        if os.path.isfile(file):
            pickle_in = open(file, 'rb')
            return pickle.load(pickle_in) # read serialized objects from file
        else:
            return None
    except FileNotFoundError:
        print("Error deserializing file: " + file_name + "\n")


# function that extends the df with the dates to store the forecast
def append_forecast_in_dataframe(df, forecast_set, seconds, field_name):
    df[field_name] = np.nan
    last_date = df.iloc[-1].name
    next_uni = last_date.timestamp() + seconds
    for forecast in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_uni)
        next_uni += seconds
        df.loc[next_date] = \
            [np.nan for _ in range(len(df.columns) - 1)] + [forecast]
