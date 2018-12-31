import numpy
import pandas
import unittest

from cramersv import CramersV

class TestCramersV(unittest.TestCase):
    def test_cramers_corrected_stat(self):
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
        expected_values = [0., .413, 0.995]

        for test_array, value in zip(test_arrays, expected_values):
            self.assertAlmostEqual(value, CramersV.cramers_corrected_stat(test_array), places=3)

    def test_produce_cramersV_table(self):
        test_frame = pandas.DataFrame({'Music Type':['Pop', 'Rock', 'Jazz', 'Classical', 'Other',
                                                     'Pop', 'Rock', 'Jazz', 'Classical', 'Other',
                                                     'Pop', 'Rock', 'Jazz', 'Classical', 'Other',
                                                     'Pop', 'Rock', 'Jazz', 'Classical', 'Other'],
                                       'Study':['Psycology', 'Psycology', 'Psycology', 'Psycology', 'Psycology',
                                                'Economics', 'Economics', 'Economics', 'Economics', 'Economics',
                                                'Law', 'Law', 'Law', 'Law', 'Law',
                                                'Other', 'Other', 'Other', 'Other', 'Other'],
                                        'Weights':[4, 4, 8, 12, 12,
                                                   4, 4, 8, 12, 12,
                                                   6, 6, 12, 18, 18,
                                                   6, 6, 12, 18, 18]})

        A = CramersV(test_frame, 'Weights')
        correlation_table = A.produce_cramersV_table()

        expected_frame = pandas.DataFrame(numpy.array([[1.,0],[0,1.]]), index=['Music Type', 'Study'], columns=['Music Type', 'Study'])
        pandas.testing.assert_frame_equal(expected_frame, correlation_table)
