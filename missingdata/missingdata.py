"""

"""

import numpy
import pandas

def count_nulls(dataset, dataset_name):
    """
    Provide information about missing values percentages in a dataset.
    
    Parameters
    ----------
    
    dataset: pandas.DataFrame. Input dataset.
    dataset_name: str. Dataset name.
    """
    
    message = '\nMissing data for {}:\n--------------------------'.format(dataset_name)
    print(message)

    for column in dataset.columns:
        Nan_indices = numpy.nonzero(dataset[column].isnull())[0]
        if len(Nan_indices):
            message = "{} = {}% missing".format(column, Nan_indices.size*100/len(dataset))
            print(message)

def fill_missing_values_by_remapping(train_set, test_set, 
                                     feature_to_fill, mapping_feature,
                                     aggregation_flag):
    """
    Perform data inputation for a given input feature by using information obtained from another one.
    Values of "feature_to_fill" are grouped by "mapping_feature", then a basic statistic estimator, 
    specified by "aggregation_flag", is computed. The resulting mapping dataframe is used to fill missing values of "feature_to_fill".
    corresponding to values of "mapping_feature".
    
    Parameters
    ----------

    train_set: pandas.DataFrame. Training dataset.
    test_set: pandas.DataFrame. Test dataset.
    feature_to_fill: str. Feature to fill label.
    mapping_feature: str. Mappggin feature name.
    aggregation_flag: str. Indicates how the remapping feature should be used to fill the missing values.
                          Allowed values are "median", "mean", "min" and "max".
    """
    
    if aggregation_flag == 'median':
        mapping_frame = train_set.groupby(mapping_feature)[feature_to_fill].median()
    elif aggregation_flag == 'mean':
        mapping_frame = train_set.groupby(mapping_feature)[feature_to_fill].mean()
    elif aggregation_flag == 'min':
        mapping_frame = train_set.groupby(mapping_feature)[feature_to_fill].min()
    elif aggregation_flag == 'max':
        mapping_frame = train_set.groupby(mapping_feature)[feature_to_fill].max()
    else:
        raise ValueError('Invalid "aggregation_flag" value: allowed values are "median", "mean", "min" and "max".')

    for dataset in [train_set, test_set]:
        null_indices = numpy.where(pandas.isnull(dataset[feature_to_fill]))
        for index in null_indices:
            dataset.loc[index,feature_to_fill] = mapping_frame[dataset.loc[index,mapping_feature]].values