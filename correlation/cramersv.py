import numpy
import pandas
import scipy.stats as ss

from correlation.correlation import CorrelationAnalyzer

class CramersV(CorrelationAnalyzer):
    def __init__(self, dataframe, weights_column=None):
        super().__init__(dataframe, weights_column)

    def produce_correlation_table(self):
        columns = self.dataframe.columns.values.tolist()
        if self.weights_column != None:
            columns.remove(self.weights_column)
        combinations = self.compute_all_combinations(columns)

        results = numpy.array([])
        for combination in combinations:
            confusion_matrix = self.create_confusion_matrix(combination)
            results = numpy.append(results, [self.cramers_corrected_stat(confusion_matrix)])

        matrix = numpy.zeros([len(columns), len(columns)])
        iup = numpy.triu_indices(len(columns))
        matrix[iup] =results
        matrix += matrix.T
        matrix /= 2.

        return pandas.DataFrame(matrix, index=columns, columns=columns)

    @staticmethod
    def cramers_corrected_stat(confusion_matrix):
        """
        Calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return numpy.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
