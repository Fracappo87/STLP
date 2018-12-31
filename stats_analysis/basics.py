#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:47:57 2018

@author: Frank

"""

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns

from IPython.display import display

def basic_count_plot(input_feature, 
                     output_feature, 
                     dataset, 
                     continuos_feature=False,
                     display_frame=False,
                     vertical = False):
    """
    Provide basic statistics (distribution plots/ count plots and crosstabs) regarding a given input
    and the corresponding output feature.
    
    Parameters
    ----------
    
    input_feature: str. Label of the input feature
    output_feature: str. Label of the output feature
    dataset: pandas.DataFrame. Input dataset.
    continuos_feature: bool. Flag to enable analysis of continuos feature.
    display_frame: bool. Flag to display results using a pandas.DataFrame
    vertical: bool. Flag to enable vertical xticks.
    """
    
    if continuos_feature:
        for class_value in dataset[output_feature].unique():
            mask=dataset[output_feature]==class_value
            sns.distplot(dataset[mask][input_feature].dropna().values, label=class_value,kde=True)
            plt.axvline(x=numpy.median(dataset[mask][input_feature].dropna().values), ymin=0, ymax=1)
        plt.legend(title=output_feature)
        plt.xlabel(input_feature )
        plt.ylabel('Frequency')
        if vertical:
            plt.xticks(rotation='vertical')
    else:
        sns.countplot(x=input_feature ,data=dataset)
        output_vs_input = pandas.crosstab(dataset[input_feature], dataset[output_feature])
        output_vs_input = output_vs_input.div(output_vs_input.sum(1).astype(float), axis=0)
        if vertical:
            plt.xticks(rotation='vertical')
        
        if display_frame:
            display(output_vs_input)

        output_vs_input.plot(kind="bar", stacked=True)
        plt.xlabel(input_feature)
        plt.ylabel('Percentage')
    plt.show()