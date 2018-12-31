import numpy
import pandas
import unittest

from ..mutual_info import MutualInfo

class TestMutualInfo(unittest.TestCase):
    def test_mutual_information_stat(self):
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
        expected_values = [0., .2755, 1.3322]

        for test_array, value in zip(test_arrays, expected_values):
            self.assertAlmostEqual(value, MutualInfo.mutual_information_stat(test_array), places=4)

    def test_produce_correlation_table(self):
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

        A = MutualInfo(test_frame, 'Weights')
        correlation_table = A.produce_correlation_table()
        mutual_info_music_type =  (120.*numpy.log(200./60)+40.*numpy.log(200./40)+40.*numpy.log(200./20))/200.
        mutual_info_study =  (80.*numpy.log(200./40) + 120.*numpy.log(200./60.))/200.
        expected_frame = pandas.DataFrame(numpy.array([[ mutual_info_music_type,0],[0, mutual_info_study]]), index=['Music Type', 'Study'], columns=['Music Type', 'Study'])
        pandas.testing.assert_frame_equal(expected_frame, correlation_table)
