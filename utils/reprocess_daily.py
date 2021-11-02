import pandas as pd
import numpy as np
from utils.ssa import SSA

def normalize_data(dataframe, mode, ignored_cols=[]):
    if mode == 'abs':
        from sklearn.preprocessing import MaxAbsScaler
        scaler_gtr = MaxAbsScaler(copy=True)  #save for retransform later
        scaler_gtr.fit(dataframe)
        data_norm = scaler_gtr.transform(dataframe)

    if mode == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler_gtr = RobustScaler(copy=True)  #save for retransform later
        scaler_gtr.fit(dataframe)
        data_norm = scaler_gtr.transform(dataframe)

    if mode == 'min_max':
        from sklearn.preprocessing import MinMaxScaler
        scaler_gtr = MinMaxScaler(feature_range=(0, 1), copy=True)  #save for retransform later
        scaler_gtr.fit(dataframe)
        data_norm = scaler_gtr.transform(dataframe)

    if mode == 'std':
        from sklearn.preprocessing import StandardScaler
        scaler_gtr = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler_gtr.fit(dataframe)
        data_norm = scaler_gtr.transform(dataframe)
    
    if ignored_cols:
        data_norm[:, ignored_cols] = dataframe[:, ignored_cols]
    
    return data_norm, scaler_gtr


def extract_data(dataframe, window_size=5, target_timstep=1, cols_x=[], cols_y=[], cols_gt=[],mode='std',ignored_cols=[]):
    '''
    The function for splitting the data
    '''
    dataframe, scaler = normalize_data(dataframe, mode, ignored_cols)

    xs = [] # return input data
    ys = [] # return output data
    ygt = [] # return groundtruth data

    if target_timstep != 1:
        for i in range(dataframe.shape[0] - window_size - target_timstep):
            xs.append(dataframe[i:i + window_size, cols_x])
            ys.append(dataframe[i + window_size:i + window_size + target_timstep,
                                cols_y])
            ygt.append(dataframe[i + window_size:i + window_size + target_timstep,
                       cols_gt])
    else:
        for i in range(dataframe.shape[0] - window_size - target_timstep):
            xs.append(dataframe[i:i + window_size, cols_x])
            ys.append(dataframe[i + window_size, cols_y])
            ygt.append(dataframe[i + window_size, cols_gt])
    return np.array(xs), np.array(ys), scaler, np.array(ygt)

def transform_ssa(input, n, sigma_lst, ignored_cols=[]):
    print("transform_ssa", input.shape)
    step = input.shape[0]
    nfeat = input.shape[-1]
    for feat in range(nfeat):
        if feat in ignored_cols:
            continue
        feat_ssa = []
        for i in range(step):
            lst_ssa = SSA(input[i, :, feat], n)
            feat_merged = lst_ssa.reconstruct(sigma_lst)
            feat_ssa.append(feat_merged)
        input[:, :, feat] = np.array(feat_ssa)

    print("transform_ssa", input.shape)
    return input
    
def ssa_extract_data(gtruth, q_ssa, h_ssa, window_size=7, target_timstep=1, mode='std'):
    '''
    generate data with separate ssa components
    '''
    gtruth, scaler_gtr = normalize_data(gtruth, mode)
    q_ssa, _ = normalize_data(q_ssa, mode)
    h_ssa, _ = normalize_data(h_ssa, mode)

    xs_q = [] # return input data
    xs_h = [] # return input data
    ygt = [] # return groundtruth data

    if target_timstep != 1:
        for i in range(gtruth.shape[0] - window_size - target_timstep):
            xs_q.append(q_ssa[i:i + window_size, :])
            xs_h.append(h_ssa[i:i + window_size, :])
            ygt.append(gtruth[i + window_size:i + window_size + target_timstep, :])
    else:
        for i in range(gtruth.shape[0] - window_size - target_timstep):
            xs_q.append(q_ssa[i:i + window_size, :])
            xs_h.append(h_ssa[i:i + window_size, :])
            ygt.append(gtruth[i + window_size, :])

    return np.array(xs_q), np.array(xs_h), scaler_gtr, np.array(ygt)

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


