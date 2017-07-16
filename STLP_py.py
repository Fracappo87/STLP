"""
Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause

"""

"""

    Submodules and relative classes:

    1) MissingData:

        allows extraction and study of features with missing data from a given dataset.

        Private parameters
        ----------
        __missings : pandas data frame built after the extraction of features with Nan values. To be used internally by the class methods
        __ifcount : boolean variable, used to check whether or not the "count" method has been called
        __trs : double variable, used to store the threshold value

        Attributes
        ----------
        n_features_ : int
        The number of features containing missing values.

        support_ : array of shape [n_features].
        The mask of selected features.

        n_features_filter_ : int.
        The number of features whose missing values percentage is higher than a given threshold.

        support_filter_ : array of shape [n_features_filter_].
        The mask of selected features whose missing value percentage is higher than a given threshold.

        Methods
        -------

        count: counts the number of features with missing values
            Parameters
            ----------
            X : Pandas Data Frame-like, shape = [n_samples, n_features]

            labels : list-like, shape = [n_features]
            List of strings characterizing the columns.

            threshold : reference value for the percentage of missing data, 0 by default. When specified, it allows the computation of                              n_features_filter_ and support_filter_

        summary: produces a summary table, containing feature name, total missing data and percentage of missing data

        transform: returns a new dataframe, where features with non null Nan-percentage have been removed
            Parameters
            ----------
            X : Pandas Data Frame-like, shape = [n_samples, n_features]
            The training input samples.


    2) CategoricalHero
    ...
"""

from MissingData import *
from CategoricalHero import *
