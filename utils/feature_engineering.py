import numpy as np
import pandas as pd

dat = pd.read_csv('./data/SonTay.csv', header=0)
dat = dat[['Q', 'H', 'evaporation', 'rainfall', 'avgtemp', 'date']]
dat['date'] = pd.to_datetime(dat.date, dayfirst=True)
dat['dayofyear'] = dat.apply(lambda x : x['date'].dayofyear, axis=1)
dat['day_sin'] = np.sin(2 * np.pi * dat['dayofyear']/366.0)
dat['day_cos'] = np.cos(2 * np.pi * dat['dayofyear']/366.0)

dat.to_csv('./data/mod_SonTay.csv')