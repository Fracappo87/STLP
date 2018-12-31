import numpy
import pandas
import unittest

from correlation import CorrelationAnalyzer

class TestCorrelationAnalyzer(unittest.TestCase):
    def test_init(self):
        self.assertRaises(ValueError, CorrelationAnalyzer, 'A')
        A = CorrelationAnalyzer(pandas.DataFrame({'A':[]}), 'B')
        self.assertEqual('B', A.weights_column)

    def test_compute_all_combinations(self):
        columns = ['A', 'B', 'C']
        combinations = CorrelationAnalyzer.compute_all_combinations(columns)
        expected_combinations = [['A', 'A'],
                                 ['A', 'B'],
                                 ['A', 'C'],
                                 ['B', 'B'],
                                 ['B', 'C'],
                                 ['C', 'C']]

        self.assertCountEqual(expected_combinations, combinations)

    def test_create_confusion_matrix(self):
        test_frame = pandas.DataFrame({'Animal':['leopard', 'dingo', 'dingo', 'crocodile'],
                                       'Species':['mammal', 'mammal', 'mammal', 'reptile'],
                                       'Size':['medium', 'small', 'small', 'large'],
                                       'Weights':[10, 2, 34, 7]})

        list_of_columns = [['Animal', 'Species'], ['Animal', 'Size'], ['Species', 'Size']]

        # Creating contingency table without using weights
        A = CorrelationAnalyzer(test_frame)
        expected_arrays = [numpy.array([[0, 1], [2, 0], [1, 0]]),
                           numpy.array([[1, 0, 0], [0, 0, 2], [0, 1, 0]]),
                           numpy.array([[0, 1, 2], [1, 0, 0]])]

        for columns, expected_array in zip(list_of_columns, expected_arrays):
            confusion_matrix = A.create_confusion_matrix(columns)
            numpy.testing.assert_array_equal(expected_array, confusion_matrix)

        # Creating contingency table by using weights
        A = CorrelationAnalyzer(test_frame, 'Weights')
        expected_arrays = [numpy.array([[0., 7.], [36., 0.], [10., 0.]]),
                           numpy.array([[7., 0., 0.], [0., 0., 36.], [0., 10., 0.]]),
                           numpy.array([[0., 10., 36.], [7., 0., 0.]])]

        for columns, expected_array in zip(list_of_columns, expected_arrays):
            confusion_matrix = A.create_confusion_matrix(columns)
            numpy.testing.assert_array_equal(expected_array, confusion_matrix)
