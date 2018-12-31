"""
"""

import numpy
import pandas

from correlation.correlation import CorrelationAnalyzer

class TheilsU(CorrelationAnalyzer):
    def __init__(self, dataframe, weights_column=None):
        super().__init__(dataframe, weights_column)

    def produce_correlation_table(self):
        columns = self.dataframe.columns.values.tolist()
        if self.weights_column != None:
            columns.remove(self.weights_column)
        combinations = self.compute_all_combinations(columns)

        new_frame = pandas.DataFrame(index=columns, columns=columns, dtype=float)
        for combination in combinations:
            confusion_matrix = self.create_confusion_matrix(combination)
            upper_value, lower_value = self.theils_u_stat(confusion_matrix)
            new_frame.loc[combination[0], combination[1]] = upper_value
            new_frame.loc[combination[1], combination[0]] = lower_value
            
        return new_frame
    
    @staticmethod
    def theils_u_stat(confusion_matrix):
        """
        Calculate Theil 's U, as described in
        https://en.wikipedia.org/wiki/Uncertainty_coefficient
        """

        marginal_X_sum = confusion_matrix.sum(axis=1).reshape(-1,1)
        marginal_Y_sum = confusion_matrix.sum(axis=0).reshape(1,-1)
        total_n_observations = confusion_matrix.sum()

        X_entropy = ((marginal_X_sum/total_n_observations)*numpy.log(total_n_observations/marginal_X_sum)).sum()
        Y_entropy = ((marginal_Y_sum/total_n_observations)*numpy.log(total_n_observations/marginal_Y_sum)).sum()
        
        zero_mask = confusion_matrix==0
        probabilities_ratio = confusion_matrix*total_n_observations/(marginal_X_sum*marginal_Y_sum)
        
        # avoid computing log(0)
        probabilities_ratio[zero_mask] = 1e-8
        pointwise_mutual_information = confusion_matrix*numpy.log(probabilities_ratio)

        pointwise_mutual_information /= total_n_observations
        mutual_information = numpy.nansum(pointwise_mutual_information)
        return mutual_information/X_entropy, mutual_information/Y_entropy

