from collections import namedtuple
from itertools import islice
import numpy as np
import pandas as pd

from dataclasses import dataclass


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


# def old_interleave_elements(start_points, end_points):
#     """ Given two equal sized arrays, produces an output array of double that size that contains elements of start_points interleaved with elements of end_points
#     Example:
#         a_starts = ['A','B','C','D']
#         a_ends = ['a','b','c','d']
#         a_interleaved = interleave_elements(a_starts, a_ends)
#         >> a_interleaved: ['A','a','B','b','C','c','D','d']
#     """
#     assert np.shape(start_points) == np.shape(end_points), f"start_points and end_points must be the same shape. np.shape(start_points): {np.shape(start_points)}, np.shape(end_points): {np.shape(end_points)}"
#     start_points = np.atleast_2d(start_points)
#     end_points = np.atleast_2d(end_points)
#     all_points_shape = (np.shape(start_points)[0] * 2, np.shape(start_points)[1]) # it's double the length of the start_points
#     all_points = np.zeros(all_points_shape)
#     all_points[np.arange(0, all_points_shape[0], 2), :] = start_points # fill the even elements
#     all_points[np.arange(1, all_points_shape[0], 2), :] = end_points # fill the odd elements
#     assert np.shape(all_points)[0] == (np.shape(start_points)[0] * 2), f"newly created all_points is not of corrrect size! np.shape(all_points): {np.shape(all_points)}"
#     return all_points

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


# def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size):
#     start = clearing_time_index + 1 - sub_window_size + 1
    
#     sub_windows = (
#         start +
#         # expand_dims are used to convert a 1D array to 2D array.
#         np.expand_dims(np.arange(sub_window_size), 0) +
#         np.expand_dims(np.arange(max_time + 1), 0).T
#     )
    
#     return array[sub_windows]


# def vectorized_stride_v2(array, clearing_time_index, max_time, sub_window_size, stride_size):
#     start = clearing_time_index + 1 - sub_window_size + 1
    
#     sub_windows = (
#         start + 
#         np.expand_dims(np.arange(sub_window_size), 0) +
#         # Create a rightmost vector as [0, V, 2V, ...].
#         np.expand_dims(np.arange(max_time + 1, step=stride_size), 0).T
#     )
    
#     return array[sub_windows]


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
#             # Try to get __dict__ from the item:
#             try:
#                 curr_value_dict_rep = vars(curr_value) # gets the .__dict__ property if curr_value has one, otherwise throws a TypeError
#                 print_keys_if_possible(f'{curr_key}.__dict__', curr_value_dict_rep, max_depth=max_depth, depth=depth, omit_curr_item_print=True) # do not increase depth in this regard so it prints at the same level. Also tell it not to print again.

#             except TypeError:
#                 # print(f"{depth_string}- {curr_value_type}")
#                 return None # terminal item

            # print(f'AttributeError: {e}')
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
   
 
# def get_closests(df, col, val):
#     """ Requires already sorted lists. """
#     lower_idx = pd.bisect_left(df[col].values, val)
#     higher_idx = pd.bisect_right(df[col].values, val)
#     if higher_idx == lower_idx:      #val is not in the list
#         return lower_idx - 1, lower_idx
#     else:                            #val is in the list
#         return lower_idx

    
# def find_closest_values(target, source, k_matches=1):
#     """[summary]
    
#     Usage:
#         find_closest_values(target, source, k_matches=1)

#     Args:
#         target ([type]): [description]
#         source ([type]): [description]
#         k_matches (int, optional): [description]. Defaults to 1.

#     Returns:
#         [type]: [description]
#     """
#     k_above = source[source >= target].nsmallest(k_matches)
#     k_below = source[source < target].nlargest(k_matches)
#     k_all = pd.concat([k_below, k_above]).sort_values()
#     return k_all




# class MatrixFlattenTransformer(object):
# """ Supposed to allow easy transformation of data from a flattened representation to the original.
# Usage:
#     trans = MatrixFlattenTransformer(original_data_shape)
#     test_all_positions_matrix = trans.unflatten(flat_all_positions_matrix)
#     print(f'np.shape(test_all_positions_matrix): {np.shape(test_all_positions_matrix)}')
# """
#     """ TODO: does not yet work. for MatrixFlattenTransformer."""
#     def __init__(self, original_data_shape):
#         super(MatrixFlattenTransformer, self).__init__()
#         self.original_data_shape = original_data_shape

#     def flatten(self, data):
#         data_shape = np.shape(data)
#         original_flat_shape = np.prod(self.original_data_shape)
#         # assert np.shape(data) == self.original_data_shape, f"data passed in to flatten (with shape {np.shape(data)}) is not equal to the original data shape: {self.original_data_shape}"
#         assert data_shape == original_flat_shape, f"data passed in to flatten (with shape {data_shape}) is not equal to the original shape's number of items (shape: {self.original_data_shape}, original_flat_shape: {original_flat_shape}"
#         return np.reshape(data, (-1, 1))
        
#     def unflatten(self, flat_data):
#         flat_data_shape = np.shape(flat_data)
#         original_data_shape_ndim = len(self.original_data_shape)
#         # assert (flat_data_shape[:original_data_shape_ndim] == self.original_data_shape), f"data passed in to unflatten (with shape {flat_data_shape}) must match the original data shape ({self.original_data_shape}), at least up to the number of dimensions in the original"
#         additional_dimensions = flat_data_shape[original_data_shape_ndim:]        
#         return np.reshape(flat_data, (self.original_data_shape[0], self.original_data_shape[1], *additional_dimensions))
        

# ==================================================================================================================== #
# Discrete Bins/Binning                                                                                                #
# ==================================================================================================================== #

@dataclass
class BinningInfo(object):
    """Docstring for BinningInfo."""
    variable_extents: tuple
    step: float
    num_bins: int
    bin_indicies: np.ndarray


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

def compute_paginated_grid_config(num_required_subplots, max_num_columns, max_subplots_per_page=None, data_indicies=None, last_figure_subplots_same_layout=True, debug_print=False):
    """ Fills row-wise first, and constrains the subplots values to just those that you need
    Args:
        num_required_subplots ([type]): [description]
        max_num_columns ([type]): [description]
        max_subplots_per_page ([type]): If None, pagination is effectively disabled and all subplots will be on a single page.
        data_indicies ([type], optional): your indicies into your original data that will also be accessible in the main loop. Defaults to None.
        last_figure_subplots_same_layout (bool): if True, the last page has the same number of items (same # columns and # rows) as the previous (full/complete) pages.
        
        
    Example:
    
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
        all_positions_matrix, flat_all_positions_matrix, original_data_shape = build_all_positions_matrix(active_one_step_decoder.xbin_centers, active_one_step_decoder.ybin_centers)
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


