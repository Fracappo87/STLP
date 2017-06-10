"""
Author: Francesco Capponi <capponi.francesco87@gmail.com>

License: BSD 3 clause

"""

from __future__ import print_function, division
import pandas as pd
import numpy as np
from scipy import stats



class MissingData(object):

    """
    Extraction and study of features with missing data from a given dataset.

    Attributes
    ----------
    n_features_ : int
    The number of features containing missing values.

    support_ : array of shape [n_features].
    The mask of selected features - only confirmed ones are True.

    n_features_filter_ : int.
    The number of features whose missing values percentage is higher than a given threshold.

    support_filter_ : array of shape [n_features_filter_].
    The mask of selected features whose missing value percentage is higher than a given threshold - only confirmed ones are True.

    summary_: pandas Dataframe of shape [n_features or n_features_filter_,3]
    The data frame has the following form

    feature name | Total # of missing data | Percentage of missing data

    filtered_: a new data set, with n_features_filter_ removed features.

    """



    def __init__(self):
        self.__missings = None
        self.__ifcount = True
        self.__trs = 0.



    def count(self, X, labels,threshold=0.):

        """
            Counts the number of features with missing values
            Parameters
            ----------
            X : Pandas Data Frame-like, shape = [n_samples, n_features]

            labels : list-like, shape = [n_features]
            List of strings characterizing the columns.

            threshold : reference value for the percentage of missing data, 0 by default. When specified, it allows the computation of                              n_features_filter_ and support_filter_
        """

        return self._count(X, labels,threshold)



    def summary(self):

        """
        Produces a summary table, containing feature name, total missing data and percentage of missing data
        Parameters
        ----------

        threshold : reference value for the percentage of missing data, 0 by default. When specified, it allows the computation of                              n_features_filter_ and support_filter_
        """

        return self._summary()



    def transform(self, X):

        """
        Return a new dataframe, where features with non null Nan-percentage have been removed
        Parameters
        ----------
        X : Pandas Data Frame-like, shape = [n_samples, n_features]
        The training input samples.

        labels : list-like, shape = [n_features]
        List of strings characterizing the columns.

        threshold : reference value for the percentage of missing data, 0 by default. When specified, it allows the computation of                              n_features_filter_ and support_filter_
        """

        return self._transform(X)

    ##############################################################################



    def __sanitycheck(self,X, labels, threshold=0.):
        # sanity check: just to be sure the user is giving the right parameters
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a dataframe!')
        elif not isinstance(labels, list):
            raise ValueError('Labels has to be a list!')
        elif not all(isinstance(s, str) for s in labels):
            raise ValueError('Labels has to be a list of strings!')
        elif not isinstance(threshold, float):
            raise ValueError('The threshold has to be a positive float number!')
        if threshold < 0 or threshold > 1:
            raise ValueError('The threshold has to be a positive float number, between 0 and 1!')



    def _count(self, X, labels, threshold=0.):

        self.__trs = threshold
        self.__sanitycheck(X, labels, threshold)

        # Counting total number of nan values for each column
        # Count percentage of nan values
        total = X[labels].isnull().sum().sort_values(ascending=False)
        percent=(X[labels].isnull().sum()/len(X[labels])).sort_values(ascending=False)

        # Define a private dataframe, to be used inside the other methods
        self.__missings = pd.concat([total, percent], axis=1,keys=['Total','Percent'])
        self.__ifcount = False # needed as a check for summary

        self.support_=list(self.__missings[(self.__missings['Percent']>0.)].index)
        self.n_features_=len(self.support_)

        if self.__trs > 0:
            self.support_filter_=list(self.__missings[(self.__missings['Percent']> self.__trs)].index)
            self.n_features_filter_=len(self.support_filter_)
        else:
            self.support_filter_=self.support_
            self.n_features_filter_=self.n_features_



    def _summary(self):

      if self.__ifcount:
          raise ValueError('You have to call the count( X, labels,threshold) method first!')
      else:
         return self.__missings[(self.__missings['Percent']>self.__trs)]



    def transform(self,X):

        if self.__ifcount:
            raise ValueError('You have to call the count( X, labels,threshold) method first!')
        else:
            if self.n_features_filter_ > 0:
                X=X.drop(self.support_filter_,1)

        return X
