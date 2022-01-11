import numpy as np
import pandas as pd


from dataclasses import dataclass


@dataclass
class BinningInfo(object):
    """Docstring for BinningInfo."""
    variable_extents: tuple
    step: float
    num_bins: int
    bin_indicies: np.ndarray
    
    
def build_spanning_bins(variable_values, max_bin_size:float, debug_print=False):
    """ out_digitized_variable_bins include both endpoints (bin edges)

    Args:
        variable_values ([type]): [description]
        max_bin_size (float): [description]
        debug_print (bool, optional): [description]. Defaults to False.

    Returns:
        out_digitized_variable_bins [type]: [description]
        out_binning_info [BinningInfo]: contains info about how the binning was conducted
    """
    # compute extents:
    curr_variable_extents = (np.nanmin(variable_values), np.nanmax(variable_values))
    num_subdivisions = int(np.ceil((curr_variable_extents[1] - curr_variable_extents[0])/max_bin_size)) # get the next integer size above float_bin_size
    actual_subdivision_step_size = (curr_variable_extents[1] - curr_variable_extents[0]) / float(num_subdivisions) # the actual exact size of the bin
    if debug_print:
        print(f'for max_bin_size: {max_bin_size} -> num_subdivisions: {num_subdivisions}, actual_subdivision_step_size: {actual_subdivision_step_size}')
    # out_bin_indicies = np.arange(num_subdivisions)
    out_binning_info = BinningInfo(curr_variable_extents, actual_subdivision_step_size, num_subdivisions, np.arange(num_subdivisions))
    out_digitized_variable_bins = np.linspace(curr_variable_extents[0], curr_variable_extents[1], num_subdivisions, dtype=float)#.astype(float)
    
    assert out_digitized_variable_bins[-1] == out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!"
    assert out_digitized_variable_bins[0] == out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!"

    # All above arge the bin_edges
    

    return out_digitized_variable_bins, out_binning_info


def get_bin_centers(bin_edges):
    """ For a series of 1D bin edges given by bin_edges, returns the center of the bins. Output will have one less element than bin_edges. """
    return (bin_edges[:-1] + np.diff(bin_edges) / 2.0)
    


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



## Pandas DataFrame helpers:
def partition(df: pd.DataFrame, partitionColumn: str):
    # splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
    unique_values = np.unique(df[partitionColumn]) # array([ 0,  1,  2,  3,  4,  7, 11, 12, 13, 14])
    grouped_df = df.groupby([partitionColumn]) #  Groups on the specified column.
    return unique_values, np.array([grouped_df.get_group(aValue) for aValue in unique_values], dtype=object) # dataframes split for each unique value in the column


