import pandas as pd
import numpy as np

def normalize_data(dataframe, mode):
    if mode == 'abs':
        from sklearn.preprocessing import MaxAbsScaler
        max_abs = MaxAbsScaler(copy=True)  #save for retransform later
        max_abs.fit(dataframe)
        data_norm = max_abs.transform(dataframe)

        return data_norm, max_abs

    if mode == 'robust':
        from sklearn.preprocessing import RobustScaler
        robust = RobustScaler(copy=True)  #save for retransform later
        robust.fit(dataframe)
        data_norm = robust.transform(dataframe)

        return data_norm, robust

    if mode == 'min_max':
        from sklearn.preprocessing import MinMaxScaler
        minmax = MinMaxScaler(feature_range=(0, 1), copy=True)  #save for retransform later
        minmax.fit(dataframe)
        data_norm = minmax.transform(dataframe)

        return data_norm, minmax
    if mode == 'std':
        from sklearn.preprocessing import StandardScaler
        stdscaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        stdscaler.fit(dataframe)
        data_norm = stdscaler.transform(dataframe)

        return data_norm, stdscaler


def extract_data(dataframe, window_size=5, target_timstep=1, cols_x=[], cols_y=[], cols_gt=[],mode='std'):
    '''
    The function for splitting the data
    '''
    dataframe, scaler = normalize_data(dataframe, mode)

    xs = [] # return input data
    ys = [] # return output data
    ygt = [] # return groundtruth data

    if target_timstep != 1:
        for i in range(dataframe.shape[0] - window_size - target_timstep):
            xs.append(dataframe[i:i + window_size, cols_x])
            ys.append(dataframe[i + window_size:i + window_size + target_timstep,
                                cols_y].reshape(target_timstep, len(cols_y)))
            ygt.append(dataframe[i + window_size:i + window_size + target_timstep,
                       cols_gt].reshape(target_timstep, len(cols_gt)))
    else:
        for i in range(dataframe.shape[0] - window_size - target_timstep):
            xs.append(dataframe[i:i + window_size, cols_x])
            ys.append(dataframe[i + window_size, cols_y])
            ygt.append(dataframe[i + window_size, cols_gt])
    return np.array(xs), np.array(ys), scaler, np.array(ygt)


def ed_extract_data(dataframe, window_size=5, target_timstep=1, cols_x=[], cols_y=[], mode='std'):
    dataframe, scaler = normalize_data(dataframe, mode)

    en_x = []
    de_x = []
    de_y = []

    for i in range(dataframe.shape[0] - window_size - target_timstep):
        en_x.append(dataframe[i:i + window_size, cols_x])

        #decoder input is q and h of 'window-size' days before
        de_x.append(dataframe[i + window_size - 1:i + window_size + target_timstep - 1,
                              cols_y].reshape(target_timstep, len(cols_y)))
        de_y.append(dataframe[i + window_size:i + window_size + target_timstep,
                              cols_y].reshape(target_timstep, len(cols_y)))

    en_x = np.array(en_x)
    de_x = np.array(de_x)
    de_y = np.array(de_y)
    de_x[:, 0, :] = 0

    return en_x, de_x, de_y, scaler

def roll_data(dataframe, cols_x, cols_y, mode='min_max'):
    dataframe, scaler = normalize_data(dataframe, mode)
    #dataframe = dataframe.drop('time', axis=1)

    X = dataframe[:, cols_x]
    y = dataframe[:, cols_y]

    return X, y, scaler


