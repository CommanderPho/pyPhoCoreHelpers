from collections import namedtuple
from itertools import islice
from typing import Callable, Optional
import numpy as np
import pandas as pd

from dataclasses import dataclass
from attrs import define, field, Factory # used for Paginator

from pyphocorehelpers.function_helpers import function_attributes

# ==================================================================================================================== #
# List-Like and Iterators                                                                                              #
# ==================================================================================================================== #

def safe_get(list, index, fallback_value):
    """Similar to dict's .get(key, fallback) function but for lists. Returns a fallback/default value if the index is not valid for the list, otherwise returns the value at that index.
    Args:
        list (_type_): a list-like object
        index (_type_): an index into the list
        fallback_value (_type_): any value to be returned when the indexing fails

    Returns:
        _type_: the value in the list, or the fallback_value is the index is not valid for the list.
    """
    try:
        return list[index]
    except IndexError:
        return fallback_value


def safe_len(v):
    """ 2023-05-08 - tries to return the length of v if possible, otherwise returns None """
    try:
        return len(v)
    except Exception as e:
        # raise e
        print(e)
        return None
    

def safe_find_index_in_list(a_list, a_search_obj):
	""" tries to find the index of `a_search_obj` in the list `a_list` 
	If found, returns the index
	If not found, returns None (instead of throwing a ValueError which is the default)
	
	Example:
		an_ax = plots.axs[2]
		safe_find_index_in_list(plots.axs, an_ax)
		# list(plots.axs).index(an_ax)
	"""
	if not isinstance(a_list, list):
		a_list = list(a_list) # convert to list
	try:
		return a_list.index(a_search_obj)
	except ValueError as e:
		# Item not found
		return None
	except Exception as e:
		raise e



def is_consecutive_no_gaps(arr, enable_debug_print=False):
    """ Checks whether a passed array/list is a series of ascending indicies without gaps
    
    arr: listlike: checks if the series is from [0, ... , len(arr)-1]
    
    Usage:
        neuron_IDXs = extracted_neuron_IDXs
        is_consecutive_no_gaps(cell_ids, neuron_IDXs)
    """
    if enable_debug_print:
        print(f'is_consecutive_no_gaps(arr: {arr})')
    comparison_correct_sequence = np.arange(len(arr)) # build a series from [0, ... , N-1]
    differing_elements = np.setdiff1d(comparison_correct_sequence, arr)
    if (len(differing_elements) > 0):
        if enable_debug_print:
            print(f'\t differing_elements: {differing_elements}')
        return False
    else:
        return True
    

def sorted_slice(a,l,r):
    start = np.searchsorted(a, l, 'left')
    end = np.searchsorted(a, r, 'right')
    return np.arange(start, end)


def chunks(iterable, size=10):
    """ Chunking

    Args:
        iterable ([type]): [description]
        size (int, optional): [description]. Defaults to 10.

    Usage:
        laps_pages = [list(chunk) for chunk in _chunks(sess.laps.lap_id, curr_num_subplots)]
    """
    iterator = iter(iterable)
    for first in iterator:    # stops when iterator is depleted
        def chunk():          # construct generator for next chunk
            yield first       # yield element from for loop
            for more in islice(iterator, size - 1):
                yield more    # yield more elements from the iterator
        yield chunk()         # in outer generator, yield next chunk


def build_pairwise_indicies(target_indicies, debug_print=False):
    """ Builds pairs of indicies from a simple list of indicies, for use in computing pairwise operations.
    
    Example:
        target_indicies = np.arange(5) # [0, 1, 2, 3, 4]
        out_pair_indicies = build_pairwise_indicies(target_indicies)
            > out_pair_indicies: [(0, 1), (1, 2), (2, 3), (3, 4)]
    
    Args:
        target_indicies ([type]): [description]
        debug_print (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
  
    Usage:
        target_indicies = np.arange(5)
        out_pair_indicies = build_pairwise_indicies(target_indicies)
        # out_pair_indicies = list(out_pair_indicies)
        # print(f'out_pair_indicies: {list(out_pair_indicies)}')


        print(f'out_pair_indicies: {list(out_pair_indicies)}')

        for i, pair in enumerate(list(out_pair_indicies)):
            # first_item_lap_idx, next_item_lap_idx
            print(f'i: {i}, pair: {pair}')
    """
    start_pairs = target_indicies[0:-1] # all but the last index
    end_pairs = target_indicies[1:] # from the second to the last index
    out_pair_indicies = list(zip(start_pairs, end_pairs)) # want to wrap in list so it isn't consumed
    if debug_print:
        print(f'target_indicies: {target_indicies}\nstart_pairs: {start_pairs}\nend_pairs: {end_pairs}')
    return out_pair_indicies


def interleave_elements(start_points, end_points, debug_print:bool=False):
    """ Given two equal sized arrays, produces an output array of double that size that contains elements of start_points interleaved with elements of end_points
    Example:
        a_starts = ['A','B','C','D']
        a_ends = ['a','b','c','d']
        a_interleaved = interleave_elements(a_starts, a_ends)
        >> a_interleaved: ['A','a','B','b','C','c','D','d']
    """
    if not isinstance(start_points, np.ndarray):
        start_points = np.array(start_points)
    if not isinstance(end_points, np.ndarray):
        end_points = np.array(end_points)
    assert np.shape(start_points) == np.shape(end_points), f"start_points and end_points must be the same shape. np.shape(start_points): {np.shape(start_points)}, np.shape(end_points): {np.shape(end_points)}"
    # Capture initial shapes to determine if np.atleast_2d changed the shapes
    start_points_initial_shape = np.shape(start_points)
    end_points_initial_shape = np.shape(end_points)
    
    # Capture initial datatypes for building the appropriate empty np.ndarray later:
    start_points_dtype = start_points.dtype # e.g. 'str32'
    end_points_dtype = end_points.dtype # e.g. 'str32'
    assert start_points_dtype == end_points_dtype, f"start_points and end_points must be the same datatype. start_points.dtype: {start_points.dtype.name}, end_points.dtype: {end_points.dtype.name}"
    start_points = np.atleast_2d(start_points)
    end_points = np.atleast_2d(end_points)
    if debug_print:
        print(f'start_points: {start_points}\nend_points: {end_points}')
        print(f'np.shape(start_points): {np.shape(start_points)}\tnp.shape(end_points): {np.shape(end_points)}') # np.shape(start_points): (1, 4)	np.shape(end_points): (1, 4)
        print(f'start_points_dtype.name: {start_points_dtype.name}\tend_points_dtype.name: {end_points_dtype.name}')
      
    if (np.shape(start_points) != start_points_initial_shape) and (np.shape(start_points)[0] == 1):
        # Shape changed after np.atleast_2d(...) which erroniously adds the newaxis to the 0th dimension. Fix by transposing:
        start_points = start_points.T
    if (np.shape(end_points) != end_points_initial_shape) and (np.shape(end_points)[0] == 1):
        # Shape changed after np.atleast_2d(...) which erroniously adds the newaxis to the 0th dimension. Fix by transposing:
        end_points = end_points.T
    if debug_print:
        print(f'POST-TRANSFORM: np.shape(start_points): {np.shape(start_points)}\tnp.shape(end_points): {np.shape(end_points)}') # POST-TRANSFORM: np.shape(start_points): (4, 1)	np.shape(end_points): (4, 1)
    all_points_shape = (np.shape(start_points)[0] * 2, np.shape(start_points)[1]) # it's double the length of the start_points
    if debug_print:
        print(f'all_points_shape: {all_points_shape}') # all_points_shape: (2, 4)
    # all_points = np.zeros(all_points_shape)
    all_points = np.empty(all_points_shape, dtype=start_points_dtype) # Create an empty array with the appropriate dtype to hold the objects
    all_points[np.arange(0, all_points_shape[0], 2), :] = start_points # fill the even elements
    all_points[np.arange(1, all_points_shape[0], 2), :] = end_points # fill the odd elements
    assert np.shape(all_points)[0] == (np.shape(start_points)[0] * 2), f"newly created all_points is not of corrrect size! np.shape(all_points): {np.shape(all_points)}"
    return np.squeeze(all_points)

# ==================================================================================================================== #
# Dictionary and Maps                                                                                                  #
# ==================================================================================================================== #

def get_dict_subset(a_dict, included_keys=None, require_all_keys=False):
    """Gets a subset of a dictionary from a list of keys (included_keys)

    Args:
        a_dict ([type]): [description]
        included_keys ([type], optional): [description]. Defaults to None.
        require_all_keys: Bool, if True, requires all keys in included_keys to be in the dictionary (a_dict)

    Returns:
        [type]: [description]
    """
    if included_keys is not None:
        if require_all_keys:
            return {included_key:a_dict[included_key] for included_key in included_keys} # filter the dictionary for only the keys specified
        else:
            out_dict = {}
            for included_key in included_keys:
                if included_key in a_dict.keys():
                    out_dict[included_key] = a_dict[included_key]
            return out_dict
    else:
        return a_dict

def validate_reverse_index_map(value_to_original_index_reverse_map, neuron_IDXs, cell_ids, debug_print=True):
    """
    Used to be called `validate_cell_IDs_to_CellIDXs_map`

    value_to_original_index_reverse_map: is a dictioanry that has any thing for its keys, but each
        Example:
            # Allows reverse indexing into the linear imported array using the original cell ID indicies:
            id_arr = [ 2  3  4  5  7  8  9 10 11 12 14 17 18 21 22 23 24 25 26 27 28 29 33 34 38 39 42 44 45 46 47 48 53 55 57 58 61 62 63 64]
            linear_flitered_ids = np.arange(len(id_arr)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
            value_to_original_index_reverse_map = dict(zip(id_arr, linear_flitered_ids))    
     
    
    Usage:
        cell_ids = extracted_cell_ids
        neuron_IDXs = extracted_neuron_IDXs
        reverse_cellID_index_map = ipcDataExplorer.active_session.neurons.reverse_cellID_index_map
        validate_reverse_index_map(reverse_cellID_index_map, cell_ids, neuron_IDXs)
    """
    if debug_print:
        print(f'\t cell_ids: {cell_ids}')
        print(f'\t neuron_IDXs: {neuron_IDXs}')
    if not is_consecutive_no_gaps(neuron_IDXs, enable_debug_print=debug_print):
        if debug_print:
            print('neuron_IDXs has gaps!')
        return False
    else:
        map_start_ids = list(value_to_original_index_reverse_map.keys()) # the cellIDs that can be mapped from
        differing_elements_ids = np.setdiff1d(map_start_ids, cell_ids)
        num_differing_ids = len(differing_elements_ids)
        map_destination_IDXs = list(value_to_original_index_reverse_map.values()) # the cellIDXs that can be mapped to.
        differing_elements_IDXs = np.setdiff1d(map_destination_IDXs, neuron_IDXs)
        num_differing_IDXs = len(differing_elements_IDXs)
        if (num_differing_IDXs > 0) or (num_differing_ids > 0):
            if debug_print:
                print(f'\t differing_elements_IDXs: {differing_elements_IDXs}')
                print(f'\t differing_elements_ids: {differing_elements_ids}')
            return False
        else:
            return True

def nested_dict_set(dic, key_list, value, create_missing=True):
    """ Allows setting the value of a nested dictionary hierarchy by drilling in with the keys in key_list, creating intermediate dictionaries if needed.
    
    Attribution and Credit:
        https://stackoverflow.com/a/49290758/9732163
    
    Usage:
        d = {}
        nested_set(d, ['person', 'address', 'city'], 'New York')
        d
        >> {'person': {'address': {'city': 'New York'}}}
    """
    d = dic
    for key in key_list[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            return dic
    if key_list[-1] in d or create_missing:
        d[key_list[-1]] = value
    return dic

def flatpaths_to_nested_dicts(flat_paths_form_dict, default_value_override='Test Value', flat_path_delimiter='.', debug_print=False):
    """ Reciprocal of nested_dicts_to_flatpaths(...)

    Args:
        flat_paths_list (_type_): _description_
        default_value_override (str, optional): _description_. Defaults to 'Test Value'.
        debug_print (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
        
    Usage:
        from pyphocorehelpers.indexing_helpers import nested_dicts_to_flatpaths, flatpaths_to_nested_dicts
        flatpaths_to_nested_dicts(['SpikeAnalysisComputations._perform_spike_burst_detection_computation',
        'ExtendedStatsComputations._perform_placefield_overlap_computation',
        'ExtendedStatsComputations._perform_firing_rate_trends_computation',
        'ExtendedStatsComputations._perform_extended_statistics_computation',
        'DefaultComputationFunctions._perform_velocity_vs_pf_density_computation',
        'DefaultComputationFunctions._perform_two_step_position_decoding_computation',
        'DefaultComputationFunctions._perform_position_decoding_computation',
        'PlacefieldComputations._perform_time_dependent_placefield_computation',
        'PlacefieldComputations._perform_baseline_placefield_computation'])


        flatpaths_to_nested_dicts(_temp_compuitations_flat_functions_list)
    
    """
    out_hierarchy_dict = {}
    if isinstance(flat_paths_form_dict, list):
        flat_paths_dict = {a_flat_path:default_value_override for a_flat_path in flat_paths_form_dict}
    else:
        flat_paths_dict = flat_paths_form_dict
        # Otherwise should already have an .items() method:
        
    # for a_flat_path in flat_paths_list:
    for a_flat_path, a_value in flat_paths_dict.items():
        key_hierarchy_list = a_flat_path.split(flat_path_delimiter)
        if debug_print:
            print(f'for item {a_flat_path}:')
        out_hierarchy_dict = nested_dict_set(out_hierarchy_dict, key_hierarchy_list, a_value, create_missing=True)
        if debug_print:
            print(f'\t out_hierarchy_dict: {out_hierarchy_dict}')
    return out_hierarchy_dict


_GLOBAL_MAX_DEPTH = 20
def nested_dicts_to_flatpaths(curr_key, curr_value, max_depth=20, depth=0, flat_path_delimiter='.', debug_print=False):
    """ Reciprocal of flatpaths_to_nested_dicts(...)

        curr_key: None to start
        curr_value: assumed to be nested_hierarchy_dict to start


    Usage:
        from pyphocorehelpers.indexing_helpers import nested_dicts_to_flatpaths, flatpaths_to_nested_dicts
        _out_flatpaths_dict = nested_dicts_to_flatpaths('', _temp_compuitations_functions_list)
        _out_flatpaths_dict

        {'SpikeAnalysisComputations._perform_spike_burst_detection_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis.SpikeAnalysisComputations._perform_spike_burst_detection_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'ExtendedStatsComputations._perform_placefield_overlap_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats.ExtendedStatsComputations._perform_placefield_overlap_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'ExtendedStatsComputations._perform_firing_rate_trends_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats.ExtendedStatsComputations._perform_firing_rate_trends_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'ExtendedStatsComputations._perform_extended_statistics_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ExtendedStats.ExtendedStatsComputations._perform_extended_statistics_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'DefaultComputationFunctions._perform_velocity_vs_pf_density_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions.DefaultComputationFunctions._perform_velocity_vs_pf_density_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'DefaultComputationFunctions._perform_two_step_position_decoding_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions.DefaultComputationFunctions._perform_two_step_position_decoding_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'DefaultComputationFunctions._perform_position_decoding_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions.DefaultComputationFunctions._perform_position_decoding_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult)>,
 'PlacefieldComputations._perform_time_dependent_placefield_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_time_dependent_placefield_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>,
 'PlacefieldComputations._perform_baseline_placefield_computation': <function pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_baseline_placefield_computation(computation_result: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult, debug_print=False)>}
 
 
        _input_test_functions_list = _temp_compuitations_functions_list
        _out_flatpaths_dict = nested_dicts_to_flatpaths('', _input_test_functions_list)
        _original_hierarchical_test_dict = flatpaths_to_nested_dicts(_out_flatpaths_dict)
        assert _original_hierarchical_test_dict == _input_test_functions_list, "ERROR, flatpaths_to_nested_dicts(nested_dicts_to_flatpaths(INPUT)) should be identity transforms (should == INPUT), but they do not!"

        
    """
    if (depth >= _GLOBAL_MAX_DEPTH):
        print(f'OVERFLOW AT DEPTH {_GLOBAL_MAX_DEPTH}!')
        raise OverflowError
    elif (depth > max_depth):
        if debug_print:
            print(f'finished at DEPTH {depth} with max_depth: {max_depth}!')
        return None
        
    else:
        if debug_print:
            curr_value_type = type(curr_value)
            print(f'curr_value_type: {curr_value_type}')
            print(f'curr_value: {curr_value}')
        
        # See if the curr_value has .items() or not.
        try:    
            child_out_dict = {}
            for (curr_child_key, curr_child_value) in curr_value.items():
                # prints the current value:
                if debug_print:
                    print(f"\t {curr_child_key} - {type(curr_child_value)}")
                # process children keys
                if curr_key is None or curr_key == '':
                    child_key_path = f'{curr_child_key}' # the curr_child_key is the top-level item
                else:
                    child_key_path = f'{curr_key}{flat_path_delimiter}{curr_child_key}'    
                curr_out = nested_dicts_to_flatpaths(child_key_path, curr_child_value, max_depth=max_depth, depth=(depth+1))
                if curr_out is not None:
                    if debug_print:
                        print(f'\t curr_out: {curr_out}')
                    child_out_dict = child_out_dict | curr_out # merge in the new dict value
                    
            return child_out_dict
            # curr_value = child_out_dict
            # is_terminal_item = False   
            
        except AttributeError as e:                

            is_terminal_item = True
    
 
        if is_terminal_item:
            # A concrete item:
            if debug_print:
                print(f'TERMINAL ITEM: ({curr_key}, {curr_value})')
            return {curr_key: curr_value}
        else:
            if debug_print:
                print(f'NON-terminal item: ({curr_key}, {curr_value})')
            return None


def apply_to_dict_values(a_dict: dict, a_callable: Callable, include_condition: Callable = None) -> dict:
    """ applies the Callable a_callable to the values of a_dict 
    e.g. 

    from pyphocorehelpers.indexing_helpers import apply_to_dict_values
    """
    if include_condition is not None:
        return {k:a_callable(v) for k, v in a_dict.items() if include_condition(k,v)}
    else:
        return {k:a_callable(v) for k, v in a_dict.items()}



# ==================================================================================================================== #
# Numpy NDArrays                                                                                                       #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['numpy', 'safe', 'indexing','list'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-08 20:06', related_items=[])
def safe_numpy_index(arr, idxs: np.ndarray):
    """ tries to prevent errors when arr is a list and idxs is a numpy array. """
    try:
        return arr[idxs] # works for arr: np.ndarray and idxs: np.ndarray
    except TypeError as e:
        # "TypeError: only integer scalar arrays can be converted to a scalar index"
        # Occurs when arr is list and idxs: np.ndarray
        if isinstance(arr, list):
            return [arr[i] for i in idxs] # returns a list object
        else:
            print(f"NotImplementedError: type(arr): {type(arr)}, type(idxs): {type(idxs)}")
            raise NotImplementedError
    except Exception as e:
        raise e
        

def safe_np_vstack(arr):
    """ a version of np.vstack that doesn't throw a ValueError on empty lists
        from pyphocorehelpers.indexing_helpers import safe_np_vstack

    """
    if len(arr)>0:
        return np.vstack(arr) # without this check np.vstack throws `ValueError: need at least one array to concatenate` for empty lists
    else:
        return np.array(arr) # ChatGPT suggests returning np.empty((0, 0))  # Create an empty array with shape (0, 0)

def safe_np_hstack(arr):
    """ a version of np.hstack that doesn't throw a ValueError on empty lists
        from pyphocorehelpers.indexing_helpers import safe_np_hstack

    """
    if len(arr)>0:
        return np.hstack(arr) # without this check np.vstack throws `ValueError: need at least one array to concatenate` for empty lists
    else:
        return np.array(arr) # ChatGPT suggests returning np.empty((0, 0))  # Create an empty array with shape (0, 0)
    

# ==================================================================================================================== #
# Pandas Dataframes                                                                                                    #
# ==================================================================================================================== #


def safe_pandas_get_group(dataframe_group, key):
    """ returns an empty dataframe if the key isn't found in the group."""
    if key in dataframe_group.groups.keys():
        return dataframe_group.get_group(key)
    else:
        original_df = dataframe_group.obj
        return original_df.drop(original_df.index)
    

## Pandas DataFrame helpers:
def partition(df: pd.DataFrame, partitionColumn: str):
    """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
    History: refactored from `pyphoplacecellanalysis.PhoPositionalData.analysis.helpers`
    """
    unique_values = np.unique(df[partitionColumn]) # array([ 0,  1,  2,  3,  4,  7, 11, 12, 13, 14])
    grouped_df = df.groupby([partitionColumn]) #  Groups on the specified column.
    return unique_values, np.array([grouped_df.get_group(aValue) for aValue in unique_values], dtype=object) # dataframes split for each unique value in the column

def partition_df(df: pd.DataFrame, partitionColumn: str):
    """ splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
    History: refactored from `pyphoplacecellanalysis.PhoPositionalData.analysis.helpers`
    """
    unique_values = np.unique(df[partitionColumn]) # array([ 0,  1,  2,  3,  4,  7, 11, 12, 13, 14])
    grouped_df = df.groupby([partitionColumn]) #  Groups on the specified column.
    return unique_values, [grouped_df.get_group(aValue) for aValue in unique_values] # dataframes split for each unique value in the column


def find_neighbours(value, df, colname):
    """Claims to be O(N)
    From https://stackoverflow.com/questions/30112202/how-do-i-find-the-closest-values-in-a-pandas-series-to-an-input-number
    
    Args:
        value ([type]): [description]
        df ([type]): [description]
        colname ([type]): [description]

    Returns:
        [type]: [description]
    """
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return [lowerneighbour_ind, upperneighbour_ind] 
    
    
    #If the series is already sorted, an efficient method of finding the indexes is by using bisect functions.
 
# ==================================================================================================================== #
# Discrete Bins/Binning                                                                                                #
# ==================================================================================================================== #

def get_bin_centers(bin_edges):
    """ For a series of 1D bin edges given by bin_edges, returns the center of the bins. Output will have one less element than bin_edges. """
    return (bin_edges[:-1] + np.diff(bin_edges) / 2.0)
    
def get_bin_edges(bin_centers):
    """ For a series of 1D bin centers given by bin_centers, returns the edges of the bins. Output will have one more element than bin_centers
        Reciprocal of get_bin_centers(bin_edges)
    """
    bin_width = float((bin_centers[1] - bin_centers[0]))
    half_bin_width = bin_width / 2.0 # TODO: assumes fixed bin width
    bin_start_edges = bin_centers - half_bin_width
    last_edge_bin = bin_centers[-1] + half_bin_width # the last edge bin is one half_bin_width beyond the last bin_center
    out = bin_start_edges.tolist()
    out.append(last_edge_bin) # append the last_edge_bin to the bins.
    return np.array(out)


@dataclass
class BinningInfo(object):
    """Docstring for BinningInfo."""
    variable_extents: tuple
    step: float
    num_bins: int
    bin_indicies: np.ndarray

class BinningContainer(object):
    """A container that allows accessing either bin_edges (self.edges) or bin_centers (self.centers) """
    edges: np.ndarray
    centers: np.ndarray
    
    edge_info: BinningInfo
    center_info: BinningInfo
    
    def __init__(self, edges: Optional[np.ndarray]=None, centers: Optional[np.ndarray]=None, edge_info: Optional[BinningInfo]=None, center_info: Optional[BinningInfo]=None):
        super(BinningContainer, self).__init__()
        assert (edges is not None) or (centers is not None) # Require either centers or edges to be provided
        if edges is not None:
            self.edges = edges
        else:
            # Compute from edges
            self.edges = get_bin_edges(centers)
            
        if centers is not None:
            self.centers = centers
        else:
            self.centers = get_bin_centers(edges)
            
            
        if edge_info is not None:
            self.edge_info = edge_info
        else:
            # Otherwise try to reverse engineer edge_info:
            self.edge_info = BinningContainer.build_edge_binning_info(self.edges)
            
        if center_info is not None:
            self.center_info = center_info
        else:
            self.center_info = BinningContainer.build_center_binning_info(self.centers, self.edge_info.variable_extents)
            
            
    @classmethod
    def build_edge_binning_info(cls, edges):
        # Otherwise try to reverse engineer edge_info            
        actual_window_size = edges[2] - edges[1]
        variable_extents = [edges[0], edges[-1]] # get first and last values as the extents
        return BinningInfo(variable_extents, actual_window_size, len(edges), np.arange(len(edges)))
    
    
    @classmethod
    def build_center_binning_info(cls, centers, variable_extents):
        # Otherwise try to reverse engineer center_info
        actual_window_size = centers[2] - centers[1]
        return BinningInfo(variable_extents, actual_window_size, len(centers), np.arange(len(centers)))
    
    def setup_from_edges(self, edges: np.ndarray, edge_info: Optional[BinningInfo]=None):
        # Set the edges first:
        self.edges = edges
        if edge_info is not None:
            self.edge_info = edge_info # valid edge_info provided, use that
        else:
            # Otherwise try to reverse engineer edge_info:
            self.edge_info = BinningContainer.build_edge_binning_info(self.edges)
            # actual_window_size = self.edges[2] - self.edges[1]
            # variable_extents = [self.edges[0], self.edges[-1]] # get first and last values as the extents
            # self.edge_info = BinningInfo(variable_extents, actual_window_size, len(self.edges), np.arange(len(self.edges)))
        
        
        ## Build the Centers from the new edges:
        self.centers = get_bin_centers(edges)
        # actual_window_size = self.centers[2] - self.centers[1]
        # self.center_info = BinningInfo(self.edge_info.variable_extents, actual_window_size, len(self.centers), np.arange(len(self.centers)))
        self.center_info = BinningContainer.build_center_binning_info(self.centers, self.edge_info.variable_extents)
            

def compute_spanning_bins(variable_values, num_bins:int=None, bin_size:float=None, variable_start_value:float=None, variable_end_value:float=None):
    """[summary]

    Args:
        variable_values ([type]): The variables to be binned, used to determine the start and end edges of the returned bins.
        num_bins (int, optional): The total number of bins to create. Defaults to None.
        bin_size (float, optional): The size of each bin. Defaults to None.
        variable_start_value (float, optional): The minimum value of the binned variable. If specified, overrides the lower binned limit instead of computing it from variable_values. Defaults to None.
        variable_end_value (float, optional): The maximum value of the binned variable. If specified, overrides the upper binned limit instead of computing it from variable_values. Defaults to None.
        debug_print (bool, optional): Whether to print debug messages. Defaults to False.

    Raises:
        ValueError: [description]

    Returns:
        np.array<float>: The computed bins
        BinningInfo: information about how the binning was performed
        
    Usage:
        ## Binning with Fixed Number of Bins:    
        xbin_edges, xbin_edges_binning_info = compute_spanning_bins(pos_df.x.to_numpy(), bin_size=active_config.computation_config.grid_bin[0]) # bin_size mode
        print(xbin_edges_binning_info)
        ## Binning with Fixed Bin Sizes:
        xbin_edges_edges, xbin_edges_binning_info = compute_spanning_bins(pos_df.x.to_numpy(), num_bins=num_bins) # num_bins mode
        print(xbin_edges_binning_info)
        
    """
    assert (num_bins is None) or (bin_size is None), 'You cannot constrain both num_bins AND bin_size. Specify only one or the other.'
    assert (num_bins is not None) or (bin_size is not None), 'You must specify either the num_bins XOR the bin_size.'
    
    if variable_start_value is not None:
        curr_variable_min_extent = variable_start_value
    else:
        curr_variable_min_extent = np.nanmin(variable_values)
        
    if variable_end_value is not None:
        curr_variable_max_extent = variable_end_value
    else:
        curr_variable_max_extent = np.nanmax(variable_values)
        
    curr_variable_extents = (curr_variable_min_extent, curr_variable_max_extent)
    
    if num_bins is not None:
        ## Binning with Fixed Number of Bins:
        mode = 'num_bins'
        xnum_bins = num_bins
        xbin, xstep = np.linspace(curr_variable_extents[0], curr_variable_extents[1], num=num_bins, retstep=True)  # binning of x position
        
    elif bin_size is not None:
        ## Binning with Fixed Bin Sizes:
        mode = 'bin_size'
        xstep = bin_size
        xbin = np.arange(curr_variable_extents[0], (curr_variable_extents[1] + xstep), xstep, )  # binning of x position
        # the interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.
        xnum_bins = len(xbin)
        
    else:
        raise ValueError
    
    return xbin, BinningInfo(curr_variable_extents, xstep, xnum_bins, np.arange(xnum_bins))
            
def compute_position_grid_size(*any_1d_series, num_bins:tuple):
    """  Computes the required bin_sizes from the required num_bins (for each dimension independently)
    Usage:
    out_grid_bin_size, out_bins, out_bins_infos = compute_position_grid_size(curr_kdiba_pipeline.sess.position.x, curr_kdiba_pipeline.sess.position.y, num_bins=(64, 64))
    active_grid_bin = tuple(out_grid_bin_size)
    print(f'active_grid_bin: {active_grid_bin}') # (3.776841861770752, 1.043326930905373)
    """
    assert (len(any_1d_series)) == len(num_bins), f'(len(other_1d_series)) must be the same length as the num_bins tuple! But (len(other_1d_series)): {(len(any_1d_series))} and len(num_bins): {len(num_bins)}!'
    num_series = len(num_bins)
    out_bins = []
    out_bins_info = []
    out_bin_grid_step_size = np.zeros((num_series,))

    for i in np.arange(num_series):
        xbins, xbin_info = compute_spanning_bins(any_1d_series[i], num_bins=num_bins[i])
        out_bins.append(xbins)
        out_bins_info.append(xbin_info)
        out_bin_grid_step_size[i] = xbin_info.step

    return out_bin_grid_step_size, out_bins, out_bins_info


def debug_print_1D_bin_infos(bins, label='bins'):
    """ prints info about the 1D bins provided 
    Usage:
        debug_print_1D_bin_infos(time_window_centers, label='time_window_centers')
        >> time_window_centers: [1211.71 1211.96 1212.21 ... 2076.96 2077.21 2077.46], count: 3464, start: 1211.7133460667683, end: 2077.4633460667683, unique_steps: [0.25]
    """
    print(f'{label}: {bins}, count: {len(bins)}, start: {bins[0]}, end: {bins[-1]}, unique_steps: {np.unique(np.diff(bins))}')


# ==================================================================================================================== #
# 2D Grids/Gridding                                                                                                          #
# ==================================================================================================================== #
RowColTuple = namedtuple('RowColTuple', 'num_rows num_columns')
PaginatedGridIndexSpecifierTuple = namedtuple('PaginatedGridIndexSpecifierTuple', 'linear_idx row_idx col_idx data_idx')
RequiredSubplotsTuple = namedtuple('RequiredSubplotsTuple', 'num_required_subplots num_columns num_rows combined_indicies')

@function_attributes(short_name='compute_paginated_grid_config', tags=['page','grid','config'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-03-30 16:47')
def compute_paginated_grid_config(num_required_subplots, max_num_columns, max_subplots_per_page=None, data_indicies=None, last_figure_subplots_same_layout=True, debug_print=False):
    """ Fills row-wise first, and constrains the subplots values to just those that you need
    Args:
        num_required_subplots ([type]): [description]
        max_num_columns ([type]): [description]
        max_subplots_per_page ([type]): If None, pagination is effectively disabled and all subplots will be on a single page.
        data_indicies ([type], optional): your indicies into your original data that will also be accessible in the main loop. Defaults to None.
        last_figure_subplots_same_layout (bool): if True, the last page has the same number of items (same # columns and # rows) as the previous (full/complete) pages.
        
        
    Example:
        from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=subplots.num_columns, max_subplots_per_page=max_subplots_per_page, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
        num_pages = len(included_combined_indicies_pages)
    
    """
    def _compute_subplots_grid_layout(num_page_required_subplots: int, page_max_num_columns: int):
        """ For a single page """
        fixed_columns = min(page_max_num_columns, num_page_required_subplots) # if there aren't enough plots to even fill up a whole row, reduce the number of columns
        needed_rows = int(np.ceil(num_page_required_subplots / fixed_columns))
        return RowColTuple(needed_rows, fixed_columns)
    
    def _compute_num_subplots(num_required_subplots: int, max_num_columns: int, data_indicies=None):
        """Computes the RequiredSubplotsTuple from the required number of subplots and the max_num_columns. We start in row[0] and begin filling to the right until we exceed max_num_columns. To avoid going over, we add a new row and continue from there.
        """
        linear_indicies = np.arange(num_required_subplots)
        if data_indicies is None:
            data_indicies = np.arange(num_required_subplots) # the data_indicies are just the same as the lienar indicies unless otherwise specified
        (total_needed_rows, fixed_columns) = _compute_subplots_grid_layout(num_required_subplots, max_num_columns) # get the result for a single page before moving on
        all_row_column_indicies = np.unravel_index(linear_indicies, (total_needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
        all_combined_indicies = [PaginatedGridIndexSpecifierTuple(linear_indicies[i], all_row_column_indicies[0][i], all_row_column_indicies[1][i], data_indicies[i]) for i in np.arange(len(linear_indicies))]
        return RequiredSubplotsTuple(num_required_subplots, fixed_columns, total_needed_rows, all_combined_indicies)

    subplot_no_pagination_configuration = _compute_num_subplots(num_required_subplots, max_num_columns=max_num_columns, data_indicies=data_indicies)
    
    # once we have the result for a single page, we paginate it using the chunks function to easily separate it into pages.
    if max_subplots_per_page is None:
        max_subplots_per_page = num_required_subplots # all subplots must fit on a single page.
    included_combined_indicies_pages = [list(chunk) for chunk in chunks(subplot_no_pagination_configuration.combined_indicies, max_subplots_per_page)]
    
    if last_figure_subplots_same_layout:
        page_grid_sizes = [RowColTuple(subplot_no_pagination_configuration.num_rows, subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]
    else:
        # If it isn't required to have the same layout as the previous (full) pages, recompute the correct number of columns for just this page. This deals with the case when not even a full row is filled.
        page_grid_sizes = [_compute_subplots_grid_layout(len(a_page), subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]

    if debug_print:
        print(f'page_grid_sizes: {page_grid_sizes}')
    return subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes

def build_spanning_grid_matrix(x_values, y_values, debug_print=False):
    """ builds a 2D matrix with entries spanning x_values across axis 0 and spanning y_values across axis 1.
        
        For example, used to build a grid of position points from xbins and ybins.
    Usage:
        from pyphocorehelpers.indexing_helpers import build_spanning_grid_matrix
        all_positions_matrix, flat_all_positions_matrix, original_data_shape = build_spanning_grid_matrix(active_one_step_decoder.xbin_centers, active_one_step_decoder.ybin_centers)
        
    Outputs:
        all_positions_matrix: a 3D matrix # .shape # (num_cols, num_rows, 2)
        flat_all_positions_matrix: a list of 2-tuples of length num_rows * num_cols
        original_data_shape: a tuple containing the shape of the original data (num_cols, num_rows)
    """
    num_rows = len(y_values)
    num_cols = len(x_values)

    original_data_shape = (num_cols, num_rows) # original_position_data_shape: (64, 29)
    if debug_print:
        print(f'original_position_data_shape: {original_data_shape}')
    x_only_matrix = np.repeat(np.expand_dims(x_values, 1).T, num_rows, axis=0).T
    # np.shape(x_only_matrix) # (29, 64)
    flat_x_only_matrix = np.reshape(x_only_matrix, (-1, 1))
    if debug_print:
        print(f'np.shape(x_only_matrix): {np.shape(x_only_matrix)}, np.shape(flat_x_only_matrix): {np.shape(flat_x_only_matrix)}') # np.shape(x_only_matrix): (64, 29), np.shape(flat_x_only_matrix): (1856, 1)
    y_only_matrix = np.repeat(np.expand_dims(y_values, 1), num_cols, axis=1).T
    # np.shape(y_only_matrix) # (29, 64)
    flat_y_only_matrix = np.reshape(y_only_matrix, (-1, 1))

    # flat_all_positions_matrix = np.array([np.append(an_x, a_y) for (an_x, a_y) in zip(flat_x_only_matrix, flat_y_only_matrix)])
    flat_all_entries_matrix = [tuple(np.append(an_x, a_y)) for (an_x, a_y) in zip(flat_x_only_matrix, flat_y_only_matrix)] # a list of position tuples (containing two elements)
    # reconsitute its shape:
    all_entries_matrix = np.reshape(flat_all_entries_matrix, (original_data_shape[0], original_data_shape[1], 2))
    if debug_print:
        print(f'np.shape(all_positions_matrix): {np.shape(all_entries_matrix)}') # np.shape(all_positions_matrix): (1856, 2) # np.shape(all_positions_matrix): (64, 29, 2)
        print(f'flat_all_positions_matrix[0]: {flat_all_entries_matrix[0]}\nall_positions_matrix[0,0,:]: {all_entries_matrix[0,0,:]}')

    return all_entries_matrix, flat_all_entries_matrix, original_data_shape


# ==================================================================================================================== #
# Pages/Pagination                                                                                                     #
# ==================================================================================================================== #

@define(slots=False)
class Paginator:
    """ 2023-05-02 - helper that allows easily creating paginated data either for batch or realtime usage. 

    Independent of any plotting technology. Just meant to hold and paginate the data.
    
    
    TODO 2023-05-02 - See also:
    ## paginated outputs for shared cells
    included_unit_indicies_pages = [[curr_included_unit_index for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in v] for page_idx, v in enumerate(included_combined_indicies_pages)] # a list of length `num_pages` containing up to 10 items

    # Can build a list of keyword arguments that will be provided to the function of interest ahead of time
    paginated_shared_cells_kwarg_list = [dict(included_unit_neuron_IDs=curr_included_unit_indicies,
        active_identifying_ctx=active_identifying_session_ctx.adding_context(collision_prefix='_batch_plot_test', display_fn_name='batch_plot_test', plot_result_set='shared', page=f'{page_idx+1}of{num_pages}', aclus=f"{curr_included_unit_indicies}"),
        fignum=f'shared_{page_idx}', fig_idx=page_idx, n_max_page_rows=n_max_page_rows) for page_idx, curr_included_unit_indicies in enumerate(included_unit_indicies_pages)]

    Example:
    
        from pyphocorehelpers.indexing_helpers import Paginator
        
        ## Provide a tuple or list containing equally sized sequences of items:
        sequencesToShow = (rr_aclus, rr_laps, rr_replays)
        a_paginator = Paginator.init_from_data(sequencesToShow, max_num_columns=1, max_subplots_per_page=20, data_indicies=None, last_figure_subplots_same_layout=False)
        # If a paginator was constructed with `sequencesToShow = (rr_aclus, rr_laps, rr_replays)`, then:
        included_page_data_indicies, included_page_data_items = a_paginator.get_page_data(page_idx=1)
        curr_page_rr_aclus, curr_page_rr_laps, curr_page_rr_replays = included_page_data_items

    Extended Example:
        from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget
        a_paginator_controller_widget = PaginationControlWidget(n_pages=a_paginator.num_pages)

    """
    sequencesToShow: tuple
    subplot_no_pagination_configuration: RequiredSubplotsTuple
    included_combined_indicies_pages: list[list[PaginatedGridIndexSpecifierTuple]]
    page_grid_sizes: list[RowColTuple]

    nItemsToShow: int = field()
    num_pages: int = field()

    ## Computed properties:
    @property
    def num_items_per_page(self):
        """The number of items displayed on each page (one number per page).
        e.g. array([20, 20, 20, 20, 20,  8])
        """
        return np.array([(num_rows * num_columns) for (num_rows, num_columns) in self.page_grid_sizes])

    @property
    def max_num_items_per_page(self):
        """The number of items on the page with the maximum number of items.
        e.g. 20
        """
        return np.max(self.num_items_per_page)

    @classmethod
    def init_from_data(cls, sequencesToShow, max_num_columns=1, max_subplots_per_page=20, data_indicies=None, last_figure_subplots_same_layout=False):
        """ creates a Paginator object from a tuple of equal length sequences using `compute_paginated_grid_config`"""
        nItemsToShow  = len(sequencesToShow[0])
        subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nItemsToShow, max_num_columns=max_num_columns, max_subplots_per_page=max_subplots_per_page, data_indicies=data_indicies, last_figure_subplots_same_layout=last_figure_subplots_same_layout)
        num_pages = len(included_combined_indicies_pages)
        
        # Build a reverse index from (page_idx: int, a_linear_index: int) -> data_index: int

        return cls(sequencesToShow=sequencesToShow, subplot_no_pagination_configuration=subplot_no_pagination_configuration, included_combined_indicies_pages=included_combined_indicies_pages, page_grid_sizes=page_grid_sizes, nItemsToShow=nItemsToShow, num_pages=num_pages)


    def get_page_data(self, page_idx: int):
        """ 
        Usage:
            # If a paginator was constructed with `sequencesToShow = (rr_aclus, rr_laps, rr_replays)`, then:
            included_page_data_indicies, included_page_data_items = a_paginator.get_page_data(page_idx=0)
            curr_page_rr_aclus, curr_page_rr_laps, curr_page_rr_replays = included_page_data_items

        """
        ## paginated outputs for shared cells
        included_page_data_indicies = np.array([curr_included_data_index for (a_linear_index, curr_row, curr_col, curr_included_data_index) in self.included_combined_indicies_pages[page_idx]]) # a list of the data indicies on this page
        included_page_data_items = tuple([safe_numpy_index(a_seq, included_page_data_indicies) for a_seq in self.sequencesToShow])
        # included_data_indicies_pages = [[curr_included_unit_index for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in v] for page_idx, v in enumerate(self.included_combined_indicies_pages)] # a list of length `num_pages` containing up to 10 items
        return included_page_data_indicies, included_page_data_items

    # def on_page_change(self, page_idx: int, page_contents_items):
    # 	""" called when the page changes. Iterates through page_contents_items to get all the appropriate contents for that page. """
    # 	sequences_subset = []

    # 	for (a_linear_index, curr_row, curr_col, curr_included_data_index) in page_contents_items:
    # 		for a_seq in sequencesToShow
    # 			a_seq[curr_included_data_index]


# ==================================================================================================================== #
# Filling sentinal values with their adjacent values                                                                   #
# ==================================================================================================================== #

def np_ffill_1D(arr: np.ndarray, debug_print=False):
    '''  'forward-fill' the nan values in array arr. 
    By that I mean replacing each nan value with the nearest valid value from the left.
    row-wise by default
    
    Example:

    Input:
        array([[  5.,  nan,  nan,   7.,   2.],
        [  3.,  nan,   1.,   8.,  nan],
        [  4.,   9.,   6.,  nan,  nan]])
       
       
    Desired Solution:
        array([[  5.,   5.,   5.,  7.,  2.],
        [  3.,   3.,   1.,  8.,  8.],
        [  4.,   9.,   6.,  6.,  6.]])
       
       
    Most efficient solution from StackOverflow as timed by author of question: https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    Solution provided by Divakar.
    
    '''
    did_pad_1D_array = False
    if (arr.ndim < 2):
        ## Pad a 1D (N,) array to (N,1) to work just like the 2D arrays.
        if debug_print:
            print(f'np_ffill_1D(arr): (arr.ndim: {arr.ndim} < 2), adding dimension...')
        # arr = arr[:, np.newaxis] # .shape: (12100, 1)
        arr = arr[np.newaxis,:] # .shape: (1, 12100) 
        did_pad_1D_array = True # indicate that we modified the 1D array
        if debug_print:
            print(f'\t new dim: {arr.ndim}, np.shape(arr): {np.shape(arr)}.')
        assert arr.ndim == 2
    mask = np.isnan(arr)
    if debug_print:
        print(f'\t np.shape(mask): {np.shape(mask)}.')

    idx = np.where(~mask, np.arange(mask.shape[1]), 0) # chooses values from the ascending value range `np.arange(mask.shape[1])` when arr is *not* np.nan, and zeros when it is nan (idx.shape: 1D: (12100,), 2D: )
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    # output should be the same shape as the input
    if debug_print:
        print(f'\t pre-squeezed output shape: {out.shape}.')

    if did_pad_1D_array:
        out = np.squeeze(out)
        if debug_print:
            print(f'\t final 1D array restored output shape: {out.shape}.')
    return out

def np_bfill_1D(arr: np.ndarray):
    """ backfills the np.nan values instead of forward filling them 
    Simple solution for bfill provided by financial_physician in comment below
    """
    return np_ffill_1D(arr[:, ::-1])[:, ::-1]
