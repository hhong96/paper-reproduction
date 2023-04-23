from pycox.datasets import metabric
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import torch
from pycox.preprocessing.discretization import (make_cuts, IdxDiscUnknownC, _values_if_series,
    DiscretizeUnknownC, Duration2Idx)
import warnings


"""
Reference: Wang, Z., & Sun, J. (2022, August 7). Survtrace: Transformers for survival analysis with competing events. University of Illinois Urbana-Champaign
"""
def load_data(config):
    '''load data, return updated configuration.
    '''
    data = config['data']
    horizons = config['horizons']
    get_target = lambda df: (df['duration'].values, df['event'].values)

    # data processing, transform all continuous data to discrete
    df = metabric.read_df()

    # evaluate the performance at the 25th, 50th and 75th event time quantile
    times = np.quantile(df["duration"][df["event"]==1.0], horizons).tolist()

    cols_categorical = ["x4", "x5", "x6", "x7"]
    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']

    df_feat = df.drop(["duration","event"],axis=1)
    df_feat_standardize = df_feat[cols_standardize] 
    df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
    df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

    # must be categorical feature ahead of numerical features!
    df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)
    
    vocab_size = 0
    for _,feat in enumerate(cols_categorical):
        df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
        vocab_size = df_feat[feat].max() + 1
            
    # get the largest duraiton time
    max_duration_idx = df["duration"].argmax()
    df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
    df_train = df_feat.drop(df_test.index)
    df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
    df_train = df_train.drop(df_val.index)

    # assign cuts
    labtrans = LabelTransform(cuts=np.array([df["duration"].min()]+times+[df["duration"].max()]))
    labtrans.fit(*get_target(df.loc[df_train.index]))
    y = labtrans.transform(*get_target(df)) # y = (discrete duration, event indicator)
    df_y_train = pd.DataFrame({"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion": y[2][df_train.index]}, index=df_train.index)
    df_y_val = pd.DataFrame({"duration": y[0][df_val.index], "event": y[1][df_val.index],  "proportion": y[2][df_val.index]}, index=df_val.index)
    df_y_test = pd.DataFrame({"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})
    

    config['labtrans'] = labtrans
    config['num_numerical_feature'] = int(len(cols_standardize))
    config['num_categorical_feature'] = int(len(cols_categorical))
    config['num_feature'] = int(len(df_train.columns))
    config['vocab_size'] = int(vocab_size)
    config['duration_index'] = labtrans.cuts
    config['out_feature'] = int(labtrans.out_features)
    
    return df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val




def pad_col(input, val=0, where='end'):
    if input.ndim != 2:
        raise ValueError("Only works for 2-D tensors.")
    pad = torch.zeros_like(input[:, :1]) + val
    return torch.cat([input, pad] if where == 'end' else [pad, input], dim=1)



class LabelTransform:
    """
    Defining time intervals (`cuts`) needed for the `PCHazard` method [1].
    One can either determine the cut points in form of passing an array to this class,
    or one can obtain cut points based on the training data.

    Arguments:
        cuts {int, array} -- Defining cut points, either the number of cuts, or the actual cut points.
    
    Keyword Arguments:
        scheme {str} -- Scheme used for discretization. Either 'equidistant' or 'quantiles'
            (default: {'equidistant})
        min_ {float} -- Starting duration (default: {0.})
        dtype {str, dtype} -- dtype of discretization.

    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf
    """
    def __init__(self, cuts, scheme='equidistant', min_=0., dtype=None):
        self._cuts = cuts
        self._scheme = scheme
        self._min = min_
        self._dtype_init = dtype
        self._predefined_cuts = False
        self.cuts = None
        if hasattr(cuts, '__iter__'):
            if type(cuts) is list:
                cuts = np.array(cuts)
            self.cuts = cuts
            self.idu = IdxDiscUnknownC(self.cuts)
            assert dtype is None, "Need `dtype` to be `None` for specified cuts"
            self._dtype = type(self.cuts[0])
            self._dtype_init = self._dtype
            self._predefined_cuts = True
        else:
            self._cuts += 1

    def fit(self, durations, events):
        # if self._predefined_cuts:
        #     warnings.warn("Calling fit method, when 'cuts' are already defined. Leaving cuts unchanged.")
        #     return self
        self._dtype = self._dtype_init
        if self._dtype is None:
            if isinstance(durations[0], np.floating):
                self._dtype = durations.dtype
            else:
                self._dtype = np.dtype('float64')
        durations = durations.astype(self._dtype)
        # self.cuts = make_cuts(self._cuts, self._scheme, durations, events, self._min, self._dtype)
        self.duc = DiscretizeUnknownC(self.cuts, right_censor=True, censor_side='right')
        self.di = Duration2Idx(self.cuts)
        return self

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        durations = durations.astype(self._dtype)
        events = _values_if_series(events)
        dur_disc, events = self.duc.transform(durations, events)
        idx_durations = self.di.transform(dur_disc)
        cut_diff = np.diff(self.cuts)
        assert (cut_diff > 0).all(), 'Cuts are not unique.'
        t_frac = 1. - (dur_disc - durations) / cut_diff[idx_durations-1]
        if idx_durations.min() == 0:
            warnings.warn("""Got event/censoring at start time. Should be removed! It is set s.t. it has no contribution to loss.""")
            t_frac[idx_durations == 0] = 0
            events[idx_durations == 0] = 0
        idx_durations = idx_durations - 1
        # get rid of -1
        idx_durations[idx_durations < 0] = 0
        return idx_durations.astype('int64'), events.astype('float32'), t_frac.astype('float32')

    @property
    def out_features(self):
        """Returns the number of output features that should be used in the torch model.
        
        Returns:
            [int] -- Number of output features.
        """
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts) - 1