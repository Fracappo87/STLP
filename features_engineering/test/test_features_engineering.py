#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:45:21 2018

@author: Frank
"""

import pandas
import unittest

from ..features_engineering import bin_feature
from ..features_engineering import produce_combined_feature

class TestFeaturesEngineering(unittest.TestCase):
    
    def setUp(self):
        
        self.dataset = pandas.DataFrame({'x1':[.1, .2, .3, 5, 7, 8, 1, 2, 3],
                                         'x2':[1, 2,  3, .5, .7, .8, 1, 2, 3],
                                         'x3':['a', 'b', 'a', 'c', 'd', 'a', 'b', 'a', 'c']})
        
        self.mapping_dictionary = {'low':[0, 1], 'medium': [1,], 'high':[4, 9]}
    
    def test_bin_feature(self):
        self.assertRaises(ValueError, bin_feature, *[self.dataset, 'x1', 'x1_binned', self.mapping_dictionary])
        
        self.mapping_dictionary['medium'] = [1, 4]
        bin_feature(self.dataset, 'x1', 'x1_binned', self.mapping_dictionary)
        bin_feature(self.dataset, 'x2', 'x2_binned', self.mapping_dictionary)
        
        expected_dataset = pandas.DataFrame({'x1':[.1, .2, .3, 5, 7, 8, 1, 2, 3],
                                             'x2':[1, 2,  3, .5, .7, .8, 1, 2, 3],
                                             'x3':['a', 'b', 'a', 'c', 'd', 'a', 'b', 'a', 'c'],
                                             'x1_binned':['low', 'low', 'low', 'high', 'high', 'high', 'medium', 'medium', 'medium'],
                                             'x2_binned':['medium', 'medium', 'medium', 'low', 'low', 'low', 'medium', 'medium', 'medium']})
    
        pandas.testing.assert_frame_equal(expected_dataset, self.dataset)
        
    def test_produce_combined_feature(self):
        self.assertRaises(ValueError, produce_combined_feature, *[self.dataset, ['x1', 'x2'], 'x1_x2', 'violet', 'product'])
        self.assertRaises(ValueError, produce_combined_feature, *[self.dataset, ['x1', 'x2'], 'x1_x2', 'numerical', 'division'])
        
        produce_combined_feature(self.dataset, ['x1', 'x2'], 'x1_x2_sum', 'numerical', 'sum')
        produce_combined_feature(self.dataset, ['x1', 'x2'], 'x1_x2_prod', 'numerical', 'product')
        produce_combined_feature(self.dataset, ['x1', 'x3'], 'x1_x3', 'categorical')
        
        expected_frame = pandas.DataFrame({'x1':[.1, .2, .3, 5, 7, 8, 1, 2, 3],
                                         'x2':[1, 2,  3, .5, .7, .8, 1, 2, 3],
                                         'x3':['a', 'b', 'a', 'c', 'd', 'a', 'b', 'a', 'c'],
                                         'x1_x2_sum':[1.1, 2.2, 3.3, 5.5, 7.7, 8.8, 2, 4, 6],
                                         'x1_x2_prod':[.1, .4, .9, 2.5, 4.9, 6.4, 1, 4, 9],
                                         'x1_x3':['0.1_a', '0.2_b', '0.3_a', '5.0_c', '7.0_d', '8.0_a', '1.0_b', '2.0_a', '3.0_c']})
        
        pandas.testing.assert_frame_equal(expected_frame, self.dataset)
  