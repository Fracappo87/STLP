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

        """

        return self._summary()



    def transform(self, X):

        """
        Return a new dataframe, where features with non null Nan-percentage have been removed
        Parameters
        ----------
        X : Pandas Data Frame-like, shape = [n_samples, n_features]
        The training input samples.

        """

        return self._transform(X)

    ##############################################################################################


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



    def _transform(self,X):

        if self.__ifcount:
            raise ValueError('You have to call the count( X, labels,threshold) method first!')
        else:
            if self.n_features_filter_ > 0:
                X=X.drop(self.support_filter_,1)

        return X

#################################################################################################

#################################################################################################


class CategoricalHero(object):

    """
    Class for dealing with categorical features belonging to a data set.

    Private parameters
    ----------

    Attributes
    ----------

    """



    def __init__(self):
        self.__ifselect = True




    def shape_detector(self,X,Y,labels):

        """

        It detects categorical features with different categorical outcomes in X and Y.
        It takes two data frames (usually the train and test set of a machine learning experiment), a list of columns and prints out those features whose outcomes differ when moving from one data frame to the other
        These information are quite useful since they allow the user to identify that subset of categorical outcomes shared both by train and test set, and use them for training and prediction procedures.

        Parameters
        ----------
        X : Pandas Data Frame-like, shape = [n_samplesX, n_features]

        Y : Pandas Data Frame-like, shape = [n_samplesY, n_features]

        labels : list-like, shape = [n_features]
        List of strings characterizing the columns.

        """

        return self._shape_detector(X,Y,labels)



    def shape_slicer(self,X,Y,labels,choice='train'):

        """

        It removes from X or Y (or both) those observations containing categories that are not present in such data sets
        It takes information from the attribute detector_: hence it has to be used after calling the shape_detector(X,Y,labels) method.

        Parameters
        ----------
        X : Pandas Data Frame-like, shape = [n_samplesX, n_features]

        Y : Pandas Data Frame-like, shape = [n_samplesY, n_features]

        labels : list-like, shape = [n_features]
        List of strings characterizing the columns

        choice : string, values = ["train","test","both"]
        "train": remove categories that appears in test set but not in train set
        "test": remove categories that appears in train set but not in test set
        "both": remove categories that appears in train set but not in test set and viceversa

        """

        return self._shape_slicer(X,Y,labels)


#################################################################################################



    def __sanitycheck(self,X, Y,labels):
        # sanity check: just to be sure the user is giving the right parameters
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a dataframe!')
        elif not isinstance(Y, pd.DataFrame):
            raise ValueError('Y has to be a dataframe!')
        elif not isinstance(labels, list):
            raise ValueError('Labels has to be a list!')
        elif not all(isinstance(s, str) for s in labels):
            raise ValueError('Labels has to be a list of strings!')



    def _shape_detector(self,X,Y,labels):

        self.__sanitycheck(X, Y, labels)
        self.__ifselect = False
        self.detected_=[]
        self.detector_={}

        for i in labels:
        # convert categorical variable into dummy, then count the number of new columns
        #BUG DETECTED! NEED TO WORK WITH COLUMNS, NOT SHAPE
            dummy1=pd.get_dummies(X[i]).shape[1]
            dummy2=pd.get_dummies(Y[i]).shape[1]
            if dummy1 != dummy2:
               # compute Pandas series with index given by the categorical outcome and
               # values given by their relative frequence
               freq1=((X[i].dropna()).value_counts()/X[i].count()).sort_values()
               freq2=((Y[i].dropna()).value_counts()/Y[i].count()).sort_values()

               # compute the percentage of missing values for a given categorical variable
               # these are relevant information, since tell the user whether or not a mismatch
               # between the categorical outcomes of the two sets could be caused by missing data.
               percent1 = (X[i].isnull().sum()/len(X[i]))
               percent2 = (Y[i].isnull().sum()/len(Y[i]))
               self.detected_.append(i)

               temp1 = pd.concat([freq1,freq2],axis=1,keys=[('train',i),('test',i)])
               temp1.loc["% missings"] = [percent1,percent2]
               self.detector_[i]=temp1



    def shape_slicer(self,X,Y,choice="train"):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a dataframe!')
        elif not isinstance(Y, pd.DataFrame):
            raise ValueError('Y has to be a dataframe!')

        if self.__ifselect:
            raise ValueError('You have to call the shape_detector( X,Y, labels) method first!')

        to_erase=[]

        if choice=="train":
            for i in self.detected_:
                idx=self.detector_[i]["test"].isnull().any(axis=1)
                toget=self.detector_[i][idx].index.values
                X=X.drop(X[X[i].isin(list(toget))].index)
        elif choice=="test":
            for i in self.detected_:
                idx=self.detector_[i]["train"].isnull().any(axis=1)
                toget=self.detector_[i][idx].index.values
                Y=Y.drop(Y[Y[i].isin(list(toget))].index)
        elif choice=="both":
            for i in self.detected_:
                idx=self.detector_[i].isnull().any(axis=1)
                toget=self.detector_[i][idx].index.values
                X=X.drop(X[X[i].isin(list(toget))].index)
                Y=Y.drop(Y[Y[i].isin(list(toget))].index)
        else:
            raise ValueError('choice can assumes only three values: "train","test" and "both"!')

        return X,Y



    #def fill_Nan_categorical(DataFrame_X,cat_columns):
    #missing_to_fill=DataFrame_X[cat_columns].loc[:, DataFrame_X[cat_columns].isnull().any()]
    #labels=missing_to_fill.columns

    #for i in labels:

        #freq=((missing_to_fill[i].dropna()).value_counts()/DataFrame_X[i].count()).sort_values()
        #catg=freq.axes
        #probs=freq.values

        #navalues=missing_to_fill[i].loc[missing_to_fill[i].isnull()]
        #nmissing=len(navalues)

        #extract=np.random.choice(np.array(catg[0]), nmissing, p=probs)
        #imputing=pd.Series(data=extract, index=navalues.axes)

        #missing_to_fill[i].fillna(imputing,axis=0,inplace=True)
    #return missing_to_fill,missing_to_fill.columns

    #def catg_to_ord(to_change_list,DataFrameX):

    #vals=[]
    #for i in to_change_list:
    #    lab=list(DataFrameX[i].value_counts().axes[0])
    #    lab.sort(key=str.lower)
    #    vals.append([i,lab])

    #sortkeyfn = key=lambda s:s[1] #grouping with respect the grading
    #vals.sort(key=sortkeyfn)

    #resulta=[]
    #for key,valuesiter in groupby(vals, key=sortkeyfn): #grouping with respect the grading
    #      keydict=dict(zip(key,range(0,len(key))))  # creating a dictionart grading -> integer
    #      resulta.append([keydict, list(v[0] for v in valuesiter)]) # creating a list with (dictionary, features with this dictionary)

    #ordo=[]
    #for i in range(0,len(resulta)):
    #    for j in resulta[i][1]:
    #        ordo.append(DataFrameX[j].map(resulta[i][0]))

    #return pd.concat(ordo,axis=1)
