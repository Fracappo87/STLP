# STLP_py
Statistical tools for lazy people

# Dependencies
* numpy
* pandas
* scipy

----------------------------------------------------------------------------------------------------------------------------

# Description
This is just a collection of classes, useful for data analysis, data cleansing and feature engineering.
Nothing of exotic, it's just a collection of tools that I have devised when doing kaggle competitions and that I use when studying data sets:)

----------------------------------------------------------------------------------------------------------------------------

# How to use
Download, import and do as you would with any other package, like pandas, numpy or scikit-learn method:

----------------------------------------------------------------------------------------------------------------------------


# class MissingData: 
  useful for extraction and study of features with missing data from a given dataset.

## Private parameters
   
   **missings**: object
   >Pandas data frame built after the extraction of features with Nan values. To be used internally by the class methods. 
   
   **ifcount**: bool
   >Boolean variable, used to check whether or not the "count" method has been called
   
   **trs**: double 
   >Variable used to store the threshold value

## Attributes

   **n\_features\_**: int
   >The number of features containing missing values.

   **support\_**: object
   >Array of shape [n_features]: the mask of selected features.

   **n\_features\_filter\_**: int.
   >The number of features whose missing values percentage is higher than a given threshold.

   **support\_filter\_**: object
   >Array of shape [n_features_filter\_]: the mask of selected features whose missing value percentage is higher than a given threshold.

 ## Methods
 
   ### count(X, labels,threshold)
   Counts the number of features with missing values.
   
   Parameters:
   
   **X** : object
   >Pandas Data Frame-like, shape = [n_samples, n_features]
   
   **labels** : object
   >List-like, shape = [n_features]: list of strings characterizing the columns.
   
   **threshold** : double
   >Reference value for the percentage of missing data, 0 by default. When specified, it allows the computation of   n_features_filter\_ and support_filter_
        
   ### summary():
   Produces a summary table, containing feature name, total missing data and percentage of missing data
   
   ### transform(X):
   Return a new dataframe, where features with non null Nan-percentage have been removed
   
   Parameters:
   
   **X** : object
   >Pandas Data Frame-like, shape = [n_samples, n_features]:the training input samples.

