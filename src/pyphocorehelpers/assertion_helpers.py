import traceback
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray
import numpy as np

class Assert:
    """ Convenince assertion helpers that print out the value that causes the assertion along with a reasonable message instead of showing nothing
    
    
    from pyphocorehelpers.assertion_helpers import Assert
        
        
    """
    @classmethod
    def path_exists(cls, path):
        """
        # Usage:
            Assert.path_exists(global_batch_result_inst_fr_file_path)
        """
        import inspect
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Extract the variable name from the caller's local variables
        var_name = [name for name, val in frame.f_locals.items() if val is path]
        # Use the first matched variable name or 'unknown' if not found
        var_name = var_name[0] if var_name else 'unknown'
        
        assert path.exists(), f"{var_name} does not exist! {var_name}: '{path}'" # Perform the assertion with detailed error message
        

    @classmethod
    def all_equal(cls, *args):
        """ Ensures all passed *args are equal in value, if it fails, it prints the actual values of each arg.
        """
        import inspect
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        
        var_name_dict = {}
        var_names_list = [name for name, val in frame.f_locals.items()]
            
        for a_equal_checkable_var in args:
            # Extract the variable name from the caller's local variables
            var_name = [name for name, val in frame.f_locals.items() if val is a_equal_checkable_var]
            # Use the first matched variable name or 'unknown' if not found
            var_name = var_name[0] if var_name else 'unknown'
            if var_name not in var_name_dict:
                var_name_dict[var_name] = a_equal_checkable_var ## turn into dictionary
            
            # assert var_name not in var_name_dict, f"var_name: {var_name} already exists in var_name_dict: {var_name_dict}"            
            # ## could append suffix like "f{var_name}[1]"
            # var_name_dict[var_name] = a_equal_checkable_var ## turn into dictionary
            
        if len(var_name_dict) == 0:
            # return True # empty arrays are all equal
            pass
        elif len(var_name_dict) == 1:
            # if only a single array, make sure it's not accidentally passed in incorrect
            reference_var = list(var_name_dict.values())[0] # Use the first array as a reference for comparison
            # assert isinstance(reference_array, (np.ndarray))
            # assert hasattr(reference_var, 'len')
            # return True # as long as imput is intended, always True
            pass
        else:
            ## It has more than two elements:
            reference_var = list(var_name_dict.values())[0] # Use the first array as a reference for comparison
            reference_val: Any = reference_var
            values_dict = {k:v for k, v in var_name_dict.items()}
            for var_name, a_val in values_dict.items():
                if a_val != reference_val:
                    assert (a_val == reference_val), f"{var_name} must be == {reference_val} but instead {var_name}: {a_val}.\nvalues_dict: {values_dict}\n{var_name}: {a_equal_checkable_var}\n" # Perform the assertion with detailed error message
            # Check equivalence for each array in the list
            # return np.all([pairwise_numpy_fn(reference_array, an_arr, **kwargs) for an_arr in list_of_arrays[1:]]) # can be used without the list comprehension just as a generator if you use all(...) instead.
            # return all(np.all(np.array_equiv(reference_array, an_arr) for an_arr in list_of_arrays[1:])) # the outer 'all(...)' is required, otherwise it returns a generator object like: `<generator object NumpyHelpers.all_array_equiv.<locals>.<genexpr> at 0x00000128E0482AC0>`


            
            
    @classmethod
    def len_equals(cls, arr_or_list, required_length: int):
        """ Ensures the length is equal to the required_length, if it fails, it prints the actual length
        """
        import inspect
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Extract the variable name from the caller's local variables
        var_name = [name for name, val in frame.f_locals.items() if val is arr_or_list]
        # Use the first matched variable name or 'unknown' if not found
        var_name = var_name[0] if var_name else 'unknown'

        assert (len(arr_or_list) == required_length), f"{var_name} must be of length {required_length} but instead len({var_name}): {len(arr_or_list)}.\n{var_name}: {arr_or_list}\n" # Perform the assertion with detailed error message

    @classmethod
    def same_length(cls, *args):
        """ Ensures all passed *args are the same length (according to len(...), if it fails, it prints the actual length of each arg.
        """
        import inspect
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        
        var_name_dict = {}
        for arr_or_list in args:
            # Extract the variable name from the caller's local variables
            var_name = [name for name, val in frame.f_locals.items() if val is arr_or_list]
            # Use the first matched variable name or 'unknown' if not found
            var_name = var_name[0] if var_name else 'unknown'
            assert var_name not in var_name_dict, f"var_name: {var_name} already exists in var_name_dict: {var_name_dict}"
            ## could append suffix like "f{var_name}[1]"
            var_name_dict[var_name] = arr_or_list ## turn into dictionary
            
        if len(var_name_dict) == 0:
            # return True # empty arrays are all equal
            pass
        elif len(var_name_dict) == 1:
            # if only a single array, make sure it's not accidentally passed in incorrect
            reference_array = list(var_name_dict.values())[0] # Use the first array as a reference for comparison
            # assert isinstance(reference_array, (np.ndarray))
            assert hasattr(reference_array, 'len')
            # return True # as long as imput is intended, always True
            pass        
        else:
            ## It has more than two elements:
            reference_array = list(var_name_dict.values())[0] # Use the first array as a reference for comparison
            reference_len: int = len(reference_array)
            lengths_dict = {k:len(v) for k, v in var_name_dict.items()}
            for var_name, a_len in lengths_dict.items():
                if a_len != reference_len:
                    assert (a_len == reference_len), f"{var_name} must be of length {reference_len} but instead len({var_name}): {a_len}.\nreference_lengths: {lengths_dict}\n{var_name}: {arr_or_list}\n" # Perform the assertion with detailed error message
            # Check equivalence for each array in the list
            # return np.all([pairwise_numpy_fn(reference_array, an_arr, **kwargs) for an_arr in list_of_arrays[1:]]) # can be used without the list comprehension just as a generator if you use all(...) instead.
            # return all(np.all(np.array_equiv(reference_array, an_arr) for an_arr in list_of_arrays[1:])) # the outer 'all(...)' is required, otherwise it returns a generator object like: `<generator object NumpyHelpers.all_array_equiv.<locals>.<genexpr> at 0x00000128E0482AC0>`




    # @classmethod
    # def _helper_all_array_generic(cls, pairwise_numpy_fn, list_of_arrays: List[NDArray], **kwargs) -> bool:
    #     """ A n-element generalization of a specified pairwise numpy function such as `np.array_equiv`
    #     Usage:
        
    #         list_of_arrays = list(xbins.values())
    #         NumpyHelpers._helper_all_array_generic(list_of_arrays=list_of_arrays)

    #     """
    #     # Input type checking
    #     if not np.all(isinstance(arr, np.ndarray) for arr in list_of_arrays):
    #         raise ValueError("All elements in 'list_of_arrays' must be NumPy arrays.")        
    
    #     if len(list_of_arrays) == 0:
    #         return True # empty arrays are all equal
    #     elif len(list_of_arrays) == 1:
    #         # if only a single array, make sure it's not accidentally passed in incorrect
    #         reference_array = list_of_arrays[0] # Use the first array as a reference for comparison
    #         assert isinstance(reference_array, np.ndarray)
    #         return True # as long as imput is intended, always True
        
    #     else:
    #         ## It has more than two elements:
    #         reference_array = list_of_arrays[0] # Use the first array as a reference for comparison
    #         # Check equivalence for each array in the list
    #         return np.all([pairwise_numpy_fn(reference_array, an_arr, **kwargs) for an_arr in list_of_arrays[1:]]) # can be used without the list comprehension just as a generator if you use all(...) instead.
    #         # return all(np.all(np.array_equiv(reference_array, an_arr) for an_arr in list_of_arrays[1:])) # the outer 'all(...)' is required, otherwise it returns a generator object like: `<generator object NumpyHelpers.all_array_equiv.<locals>.<genexpr> at 0x00000128E0482AC0>`


    # @classmethod
    # def all_array_equal(cls, list_of_arrays: List[NDArray], equal_nan=True) -> bool:
    #     """ A n-element generalization of `np.array_equal`
    #     Usage:
        
    #         list_of_arrays = list(xbins.values())
    #         NumpyHelpers.all_array_equal(list_of_arrays=list_of_arrays)

    #     """
    #     return cls._helper_all_array_generic(np.array_equal, list_of_arrays=list_of_arrays, equal_nan=equal_nan)
    
    # @classmethod
    # def all_array_equiv(cls, list_of_arrays: List[NDArray]) -> bool:
    #     """ A n-element generalization of `np.array_equiv`
    #     Usage:
        
    #         list_of_arrays = list(xbins.values())
    #         NumpyHelpers.all_array_equiv(list_of_arrays=list_of_arrays)

    #     """
    #     return cls._helper_all_array_generic(np.array_equiv, list_of_arrays=list_of_arrays)


    # @classmethod
    # def all_allclose(cls, list_of_arrays: List[NDArray], rtol:float=1.e-5, atol:float=1.e-8, equal_nan:bool=True) -> bool:
    #     """ A n-element generalization of `np.allclose`
    #     Usage:
        
    #         list_of_arrays = list(xbins.values())
    #         NumpyHelpers.all_allclose(list_of_arrays=list_of_arrays)

    #     """
    #     return cls._helper_all_array_generic(np.allclose, list_of_arrays=list_of_arrays, rtol=rtol, atol=atol, equal_nan=equal_nan)
    
    
