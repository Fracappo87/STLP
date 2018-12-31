import numpy
import pandas
import scipy.stats as ss

from correlation.correlation import CorrelationAnalyzer

class MutualInfo(CorrelationAnalyzer):
    def __init__(self, dataframe, weights_column=None):
        super().__init__(dataframe, weights_column)

    def produce_mutual_information_table(self):
        columns = self.dataframe.columns.values.tolist()
        columns.remove(self.weights_column)
        combinations = self.compute_all_combinations(columns)

        results = numpy.array([])
        for combination in combinations:
            confusion_matrix = self.create_confusion_matrix(combination)
            results = numpy.append(results, [self.mutual_information_stat(confusion_matrix)])

        matrix = numpy.zeros([len(columns), len(columns)])
        iup = numpy.triu_indices(len(columns))
        matrix[iup] =results
        matrix += matrix.T
        matrix /= 2.

        return pandas.DataFrame(matrix, index=columns, columns=columns)

    @staticmethod
    def mutual_information_stat(confusion_matrix):
        """
        Calculate mutual information statistics, as described in
        http://www.scholarpedia.org/article/Mutual_information
        """

        marginal_X_sum = confusion_matrix.sum(axis=1).reshape(-1,1)
        marginal_Y_sum = confusion_matrix.sum(axis=0).reshape(1,-1)
        total_n_observations = confusion_matrix.sum()

        zero_mask = confusion_matrix==0
        probabilities_ratio = confusion_matrix*total_n_observations/(marginal_X_sum*marginal_Y_sum)
        # avoid computing log(0)
        probabilities_ratio[zero_mask] = 1e-8
        pointwise_mutual_information = confusion_matrix*numpy.log(probabilities_ratio)

        pointwise_mutual_information /= total_n_observations
        return numpy.nansum(pointwise_mutual_information)
