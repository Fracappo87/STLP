"""
"""

import numpy
import pandas
import unittest

from ..theilsu import TheilsU

class TestTheilsU(unittest.TestCase):
    def test_theils_u_stat(self):
        test_arrays = [numpy.array([[4, 4, 8, 12, 12],
                                    [4, 4, 8, 12, 12],
                                    [6, 6, 12, 18, 18],
                                    [6, 6, 12, 18, 18]]),
                       numpy.array([[12, 12, 8, 8, 0],
                                    [4, 4, 20, 6, 6],
                                    [2, 2, 8, 30, 18],
                                    [2, 2, 4, 16, 36]]),
                       numpy.array([[20, 0, 0, 0],
                                    [20, 0, 0, 0],
                                    [0, 20, 0, 0],
                                    [0, 0, 20, 0],
                                    [0, 0, 0, 20]])]
    
        X_entropies = [(80.*numpy.log(200./40.) + 120.*numpy.log(200./60.))/200.,
                       (80.*numpy.log(200./40.) + 120.*numpy.log(200./60.))/200.,
                       100.*numpy.log(100./20.)/100.]
        Y_entropies = [(120.*numpy.log(200./60.) + 40.*numpy.log(200./40.) + 40.*numpy.log(200./20.))/200.,
                       (40.*numpy.log(200./20.) + 40.*numpy.log(200./40.) + 120.*numpy.log(200./60.))/200.,
                       (40.*numpy.log(100./40.) + 60.*numpy.log(100./20.))/100.]
        expected_values = [0., .2755, 1.3322]

        for test_array, values, X_entropy, Y_entropy in zip(test_arrays, expected_values, X_entropies, Y_entropies):
            u_x_given_y, u_y_given_x = TheilsU.theils_u_stat(test_array)
            self.assertAlmostEqual(values/X_entropy, u_x_given_y, places=4)
            self.assertAlmostEqual(values/Y_entropy, u_y_given_x, places=4)

    def test_produce_correlation_table(self):
        test_frame = pandas.DataFrame({'Music Type':['Pop', 'Rock', 'Jazz', 'Classical', 'Other',
                                                     'Pop', 'Rock', 'Jazz', 'Classical', 'Other',
                                                     'Pop', 'Rock', 'Jazz', 'Classical', 'Other',
                                                     'Pop', 'Rock', 'Jazz', 'Classical', 'Other'],
                                       'Study':['Psycology', 'Psycology', 'Psycology', 'Psycology', 'Psycology',
                                                'Economics', 'Economics', 'Economics', 'Economics', 'Economics',
                                                'Law', 'Law', 'Law', 'Law', 'Law',
                                                'Other', 'Other', 'Other', 'Other', 'Other'],
                                        'Weights':[4, 4, 8, 12, 12,     # X marginal sums: 40
                                                   4, 4, 8, 12, 12,     #                  40
                                                   6, 6, 12, 18, 18,    #                  60
                                                   6, 6, 12, 18, 18]})  #                  60
                                # Y marginal sums 20,20, 40, 60, 60

        A = TheilsU(test_frame, 'Weights')
        correlation_table = A.produce_correlation_table()

        expected_frame = pandas.DataFrame(numpy.array([[1.,0],[0,1.]]), index=['Music Type', 'Study'], columns=['Music Type', 'Study'])
        pandas.testing.assert_frame_equal(expected_frame, correlation_table)

