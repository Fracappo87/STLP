import numpy

def generate_new_range(value, n_points, delta=.5):
    """
    Generate a range of numbers centered around an input value.
    The generated range spans from value*(1-delta) to value*(1+delta)

    Parameters
    ----------

    value: float.
         Central value
    n_points: int. 
         Number of points to be generated.
    delta: float. Parameter to determine the radius of the interval, must be between 0 and 1.

    Return
    ------
    
    new_range: numpy.ndarray. Range of values.
    """

    if delta>1 or delta <0:
        raise ValueError('Value of delta must be between 0 and 1.')
    
    lower = value*(1-.5)
    upper = value*(1+.5)
    return numpy.linspace(lower, upper, n_points)
