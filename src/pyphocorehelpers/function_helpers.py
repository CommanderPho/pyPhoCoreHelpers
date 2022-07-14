import sys
import traceback
from functools import reduce
from itertools import accumulate
from functools import wraps # This convenience func preserves name and docstring
from typing import List, Callable # for function composition

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

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




class CapturedException(DynamicParameters):
    """ An exception and its related info/context during the process of executing composed functions with error handling."""
    
    def __repr__(self):
        # Don't print out captured_result_state (as it's huge and clogs the console)
        return f'!! {self.exc} ::::: {self.exc_info}'
        # return super().__repr__()

    
    def __init__(self, exc, exc_info, captured_result_state):
        super(CapturedException, self).__init__(exc=exc, exc_info=exc_info, captured_result_state=captured_result_state)
        # self.exc = exc
        # self.exc_info = exc_info
        # self.captured_result_state = captured_result_state

    

def compose_functions_with_error_handling(*args):
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
        result = x # initially set the result to the input
        accumulated_errors = dict() # empty list for keeping track of exceptions
        for f in reversed(args):
            try:
                temp_result = f(result) # evaluate the function 'f' using the result provided from the previous output or the initial input
            except (TypeError, ValueError, NameError, AttributeError, KeyError, NotImplementedError) as e:
                exception_info = sys.exc_info()
                accumulated_errors[f] = CapturedException(e, exception_info, result)
                # accumulated_errors.append(e) # add the error to the accumulated error array
                temp_result = result # restore the result from prior to the calculations?
                # result shouldn't be updated unless there wasn't an error, so it should be fine to move on to the next function
            else:
                # only if no error occured do we commit the temp_result to result
                result = temp_result
            # finally:
            #     # do this no matter what
            #     # result
            #     pass
        return result, accumulated_errors # new function returns both the result and the accumulated errors
    return _




# Peak Finding:
# https://plotly.com/python/peak-finding/




def add_method(cls):
    """Enables post-hoc method adding to any python class using a simple decorator design. Enables easily "extending" classes at runtime or in jupyter notebooks without magic.
        From https://gist.github.com/mgarod/09aa9c3d8a52a980bd4d738e52e5b97a
        Credit to mgarod
        From article https://mgarod.medium.com/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
        
    Usage: Example of adding two functions to a class named "cls_A"
        # Non-decorator way (note the function must accept self)
        # def foo(self):
        #     print('hello world!')
        # setattr(cls_A, 'foo', foo)

        # def bar(self, s):
        #     print(f'Message: {s}')
        # setattr(cls_A, 'bar', bar)

        # Decorator can be written to take normal functions and make them methods
        @add_method(cls_A)
        def foo():
            print('hello world!')

        @add_method(cls_A)
        def bar(s):
            print(f'Message: {s}')



    """
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator
