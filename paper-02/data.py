import pandas as pd
import numpy as np
from pycox.datasets import metabric
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import STConfig
import warnings
from pycox.preprocessing.discretization import (IdxDiscUnknownC, _values_if_series, DiscretizeUnknownC, Duration2Idx)

from itables import init_notebook_mode
init_notebook_mode(all_interactive=True)


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def data_processing():
    config = STConfig
    df = metabric.read_df()

    cols_cat = ["x4", "x5", "x6", "x7"]
    cols_num = ['x0', 'x1', 'x2', 'x3', 'x8']
    times = df[df["event"] == 1]["duration"].quantile([0.25, 0.5, 0.75], interpolation="nearest").tolist()

    features = df[cols_cat]
    nums = df[cols_num]
    labels = df.drop(features, axis = 1).drop(nums, axis = 1)

    scaled_nums = StandardScaler().fit_transform(nums)
    scaled_nums = pd.DataFrame(scaled_nums, index=nums.index, columns=nums.columns)

    num_features = 0
    for idx, col in enumerate(cols_cat):
        features[col] = LabelEncoder().fit_transform(features[col]) + num_features
        num_features += len(features[col].drop_duplicates())

    features = features.astype("Float64")
    
    df = features.merge(nums, how = "left", left_index=True, right_index=True)
    # df = df.merge(labels, how="left", left_index=True, right_index=True)
    
    xtrain, xtest, ytrain, ytest = train_test_split(df, labels, test_size = 0.2, train_size = 0.8, random_state = 1)
    xtrain, xval, ytrain, yval = train_test_split(df, labels, test_size = 0.1, train_size = 0.9, random_state = 1)

    times.insert(0, labels["duration"].min())
    times.append(labels["duration"].max())

    lt = LabelTransform(cuts=times)
    lt.fit(ytrain["duration"], ytrain["event"])
    y = lt.transform(labels["duration"], labels["event"])

    cols_label = ["duration", "event", "proportion"]
    labels = pd.DataFrame({cols_label[0]: y[0], cols_label[1]: y[1], cols_label[2]: y[2]}, index=labels.index)

    df = df.merge(labels, how="inner", left_index=True, right_index=True)
    train = xtrain.merge(labels, how="inner", left_index=True, right_index=True)
    val = xval.merge(labels, how="inner", left_index=True, right_index=True)
    test = xtest.merge(labels, how="inner", left_index=True, right_index=True)

    # xtrain = train[cols_cat + cols_num].astype("Float64")
    ytrain = train[cols_label].astype("Float64")
    # xval = val[cols_cat + cols_num].astype("Float64")
    yval = val[cols_label].astype("Float64")
    # xtest = test[cols_cat + cols_num].astype("Float64")
    ytest = test[cols_label].astype("Float64")

    return df, train, ytrain, test, ytest, val, yval




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

if __name__ == "__main__":
    df, train, ytrain, test, ytest, val, yval = data_processing()

