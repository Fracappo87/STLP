import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn as sns

class CorrelationAnalyzer(object):
    def __init__(self, dataframe, weights_column=None):
        self.dataframe = dataframe
        if not isinstance(self.dataframe, pandas.DataFrame):
            raise ValueError('Expecting a pandas DataFrame to instatiate class')

        self.weights_column = weights_column

    @staticmethod
    def produce_heatmap(correlation_table):
        sns.heatmap(correlation_table)
        plt.show()

    @staticmethod
    def compute_all_combinations(input_list):
        combinations = []
        for index in range(len(input_list)):
            for jndex in range(index, len(input_list)):
                combinations.append([input_list[index], input_list[jndex]])

        return combinations

    def create_confusion_matrix(self, columns):
        """
        Create a confusion matrix for a couple fo input features
        """
        if len(columns) !=2:
            raise ValueError('Two input columns needed to perform analysis')

        if self.weights_column == None:
            return pandas.crosstab(self.dataframe[columns[0]],
                                   self.dataframe[columns[1]]).values
        else:
            confusion_matrix = pandas.crosstab(self.dataframe[columns[0]],
                                                self.dataframe[columns[1]],
                                                values=self.dataframe[self.weights_column],
                                                aggfunc=numpy.sum)
            confusion_matrix.fillna(0, inplace=True)
            return confusion_matrix.values
