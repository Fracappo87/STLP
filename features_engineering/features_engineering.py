"""

"""

import numpy

def bin_feature(dataset, input_column_name, new_column_name, mapping_dictionary):
    """
    Bin a continuos input feature into a finite numebr of bins. The new fature
    gets added to the original dataframe as a new column.
    
    Parameters
    ----------
    
    dataset: pandas.DataFrame. Input dataset.
    input_column_name: str. Name of the continuos input feature.
    new_column_name: str. Name of the binned input feature.
    mapping_dictionary: dict. Dictionary of the form
        {'key_1':[lower_1, upper_1],
                ... ... ...
         'key_n':[lower_n, upper_n]}
    where [lower_i, upper_i] represents a binning interval.
    """
    
    dataset[new_column_name] = numpy.nan
    for key, value in mapping_dictionary.items():
        if len(value)!=2:
            raise ValueError('Mapping dictionary values should be iterables of len 2.')
        else:
            mask = (dataset[input_column_name]>= value[0]) & (dataset[input_column_name]<value[1])
            dataset.loc[mask,new_column_name] = key

def produce_combined_feature(dataset, 
                             features_to_combine, new_feature_name, 
                             feature_type,
                             creation_flag='product'):
    """
    Create a new input feature by using a subset of the original inout feature space.
    
    Parameters
    ----------

    dataset: pandas.DataFrame. Input dataset.
    features_to_combine: list. List of labels referring to the features to combine.
    new_feature_name: str. Name referring to the newly created inpu feature.
    feature_type: str. Specifies the type of input feature type: allowed values are "numerical" and "categorical".
    creation_flag: str. Defines which type of operation has to be performed to create the new input feature. Allowed values are
                        "sum", "product".
    """
    
    if feature_type not in ['numerical', 'categorical']:
        raise ValueError('Invalid "feature_type" value.\nAllowed values are "numerical" and "categorical".')
    elif creation_flag not in ['sum', 'product']:
        raise ValueError('Invalid "creation_flag" value.\nAllowed values are "sum", "product".')

    if feature_type == 'numerical':
        if creation_flag == 'sum':
            dataset[new_feature_name] = dataset[features_to_combine].sum(axis=1)
        elif creation_flag == 'product':
            dataset[new_feature_name] = dataset[features_to_combine].prod(axis=1)
    elif feature_type == 'categorical':
        dataset[new_feature_name] = dataset[features_to_combine[0]].astype(str)
        for feature in features_to_combine[1:]:
            dataset[new_feature_name] = dataset[new_feature_name] + '_' + dataset[feature].astype(str)