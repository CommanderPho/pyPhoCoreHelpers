from functools import reduce
from itertools import accumulate
from typing import List, Callable # for function composition
from scipy.signal import find_peaks # peak-finding version 1
from scipy.signal import argrelextrema # peak-finding version 2


def compose_functions(*args):
    """ Composes n functions passed as input arguments into a single lambda function efficienctly.
    right-to-left ordering (default): compose(f1, f2, ..., fn) == lambda x: f1(...(f2(fn(x))...)
    # OLD: left-to-right ordering: compose(f1, f2, ..., fn) == lambda x: fn(...(f2(f1(x))...)
    Note that functions are composed from right-to-left, meaning that the first function input is the outermost function
    Usage:
        post_load_functions = [lambda a_loaded_sess: estimation_session_laps(a_loaded_sess), lambda a_loaded_sess: a_loaded_sess.filtered_by_neuron_type('pyramidal')]
    composed_post_load_function = compose_functions(*post_load_functions) # functions are composed right-to-left (math order)
    composed_post_load_function(curr_kdiba_pipeline.sess)
    """
    def _(x):
        result = x
        for f in reversed(args):
            result = f(result)
        return result
    return _



# Peak Finding:
# https://plotly.com/python/peak-finding/




