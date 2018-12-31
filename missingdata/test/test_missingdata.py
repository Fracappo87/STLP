
"""
"""

import numpy
import pandas
import unittest

from ..missingdata import fill_missing_values_by_remapping

class TestFillMissingValues(unittest.TestCase):
    
    def setUp(self):
        self.train_set = pandas.DataFrame({'A':[.1, .3, numpy.nan, 10, .4, numpy.nan, numpy.nan, 3, 1], 
                                           'B':[.4, numpy.nan, .4, 10, .4, numpy.nan, 3, numpy.nan, 1], 
                                           'C':['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'] })
    
        self.test_set = pandas.DataFrame({'A':[numpy.nan, numpy.nan, numpy.nan], 
                                       'B':[numpy.nan, numpy.nan, numpy.nan], 
                                       'C':['a', 'b', 'c'] })
        
        self.input_dictionary = {'train_set': self.train_set,
                                       'test_set': self.test_set,
                                       'feature_to_fill':'A',
                                       'mapping_feature':'C',
                                       'aggregation_flag':'ajhdjsah'}
    
    def test_fill_missing_values_by_remapping(self):
        self.assertRaises(ValueError, fill_missing_values_by_remapping,**self.input_dictionary)
        
        self.input_dictionary['aggregation_flag'] = 'mean'
        fill_missing_values_by_remapping(**self.input_dictionary)
        
        self.input_dictionary['aggregation_flag'] = 'min'
        self.input_dictionary['feature_to_fill'] = 'B'
        
        fill_missing_values_by_remapping(**self.input_dictionary)
        
        expected_train_set = pandas.DataFrame({'A':[.1, .3, .2, 10, .4, 5.2, 2, 3, 1], 
                                           'B':[.4, .4, .4, 10, .4, .4, 3, 1, 1], 
                                           'C':['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'] })
        expected_test_set = pandas.DataFrame({'A':[.2, 5.2, 2], 
                                              'B':[.4, .4, 1], 
                                              'C':['a', 'b', 'c'] })
            
        for expected, filled in zip([expected_train_set, expected_test_set],
                                    [self.train_set, self.test_set]):
            pandas.testing.assert_frame_equal(expected, filled)