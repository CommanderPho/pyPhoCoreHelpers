from collections import namedtuple
from itertools import islice
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
        




def build_spanning_bins(variable_values, max_bin_size:float, debug_print=False):
    """ DEPRICATED! out_digitized_variable_bins include both endpoints (bin edges)

    Args:
        variable_values ([type]): [description]
        max_bin_size (float): [description]
        debug_print (bool, optional): [description]. Defaults to False.

    Returns:
        out_digitized_variable_bins [type]: [description]
        out_binning_info [BinningInfo]: contains info about how the binning was conducted
    """
    raise DeprecationWarning
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




def compute_spanning_bins(variable_values, num_bins:int=None, bin_size:float=None):
    """[summary]

    Args:
        variable_values ([type]): [description]
        num_bins (int, optional): [description]. Defaults to None.
        bin_size (float, optional): [description]. Defaults to None.
        debug_print (bool, optional): [description]. Defaults to False.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
        
    Usage:
        ## Binning with Fixed Number of Bins:    
        xbin, ybin, bin_info = compute_spanning_bins(pos_df.x.to_numpy(), bin_size=active_config.computation_config.grid_bin[0]) # bin_size mode
        print(bin_info)
        ## Binning with Fixed Bin Sizes:
        xbin, ybin, bin_info = compute_spanning_bins(pos_df.x.to_numpy(), num_bins=num_bins) # num_bins mode
        print(bin_info)
        
    """
    assert (num_bins is None) or (bin_size is None), 'You cannot constrain both num_bins AND bin_size. Specify only one or the other.'
    assert (num_bins is not None) or (bin_size is not None), 'You must specify either the num_bins XOR the bin_size.'
    curr_variable_extents = (np.nanmin(variable_values), np.nanmax(variable_values))
    
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
    """ TODO: CHECK
    For a series of 1D bin centers given by bin_centers, returns the edges of the bins.
    Reciprocal of get_bin_centers(bin_edges)
    """
    half_bin_width = float((bin_centers[1] - bin_centers[0])) / 2.0 # TODO: assumes fixed bin width
    bin_starts = bin_centers - half_bin_width
    bin_ends = bin_centers + half_bin_width
    return interleave_elements(bin_starts, bin_ends)


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


def interleave_elements(start_points, end_points):
    """ Given two equal sized arrays, produces an output array of double that size that contains elements of start_points interleaved with elements of end_points
    Example:
        a_starts = ['A','B','C','D']
        a_ends = ['a','b','c','d']
        a_interleaved = interleave_elements(a_starts, a_ends)
        >> a_interleaved: ['A','a','B','b','C','c','D','d']
    """
    assert np.shape(start_points) == np.shape(end_points), f"start_points and end_points must be the same shape. np.shape(start_points): {np.shape(start_points)}, np.shape(end_points): {np.shape(end_points)}"
    start_points = np.atleast_2d(start_points)
    end_points = np.atleast_2d(end_points)
    all_points_shape = (np.shape(start_points)[0] * 2, np.shape(start_points)[1]) # it's double the length of the start_points
    all_points = np.zeros(all_points_shape)
    all_points[np.arange(0, all_points_shape[0], 2), :] = start_points # fill the even elements
    all_points[np.arange(1, all_points_shape[0], 2), :] = end_points # fill the odd elements
    assert np.shape(all_points)[0] == (np.shape(start_points)[0] * 2), f"newly created all_points is not of corrrect size! np.shape(all_points): {np.shape(all_points)}"
    return all_points



def get_dict_subset(a_dict, included_keys=None):
    """Gets a subset of a dictionary from a list of keys (included_keys)

    Args:
        a_dict ([type]): [description]
        included_keys ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if included_keys is not None:
        return {included_key:a_dict[included_key] for included_key in included_keys} # filter the dictionary for only the keys specified
    else:
        return a_dict




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


def sorted_slice(a,l,r):
    start = np.searchsorted(a, l, 'left')
    end = np.searchsorted(a, r, 'right')
    return np.arange(start, end)




## Pandas DataFrame helpers:
def partition(df: pd.DataFrame, partitionColumn: str):
    # splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
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


RowColTuple = namedtuple('RowColTuple', 'num_rows num_columns')
PaginatedGridIndexSpecifierTuple = namedtuple('PaginatedGridIndexSpecifierTuple', 'linear_idx row_idx col_idx data_idx')
RequiredSubplotsTuple = namedtuple('RequiredSubplotsTuple', 'num_required_subplots num_columns num_rows combined_indicies')

def compute_paginated_grid_config(num_required_subplots, max_num_columns, max_subplots_per_page, data_indicies=None, last_figure_subplots_same_layout=True, debug_print=False):
    """ Fills row-wise first 
    Args:
        num_required_subplots ([type]): [description]
        max_num_columns ([type]): [description]
        max_subplots_per_page ([type]): [description]
        data_indicies ([type], optional): your indicies into your original data that will also be accessible in the main loop. Defaults to None.
    """
    
    def _compute_subplots_grid_layout(num_page_required_subplots, page_max_num_columns):
        """ For a single page """
        fixed_columns = min(page_max_num_columns, num_page_required_subplots) # if there aren't enough plots to even fill up a whole row, reduce the number of columns
        needed_rows = int(np.ceil(num_page_required_subplots / fixed_columns))
        return RowColTuple(needed_rows, fixed_columns)
    
    def _compute_num_subplots(num_required_subplots, max_num_columns, data_indicies=None):
        linear_indicies = np.arange(num_required_subplots)
        if data_indicies is None:
            data_indicies = np.arange(num_required_subplots) # the data_indicies are just the same as the lienar indicies unless otherwise specified
        (total_needed_rows, fixed_columns) = _compute_subplots_grid_layout(num_required_subplots, max_num_columns)
        all_row_column_indicies = np.unravel_index(linear_indicies, (total_needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
        all_combined_indicies = [PaginatedGridIndexSpecifierTuple(linear_indicies[i], all_row_column_indicies[0][i], all_row_column_indicies[1][i], data_indicies[i]) for i in np.arange(len(linear_indicies))]
        return RequiredSubplotsTuple(num_required_subplots, fixed_columns, total_needed_rows, all_combined_indicies)

    subplot_no_pagination_configuration = _compute_num_subplots(num_required_subplots, max_num_columns=max_num_columns, data_indicies=data_indicies)
    included_combined_indicies_pages = [list(chunk) for chunk in chunks(subplot_no_pagination_configuration.combined_indicies, max_subplots_per_page)]
    
    if last_figure_subplots_same_layout:
        page_grid_sizes = [RowColTuple(subplot_no_pagination_configuration.num_rows, subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]
        
    else:
        page_grid_sizes = [_compute_subplots_grid_layout(len(a_page), subplot_no_pagination_configuration.num_columns) for a_page in included_combined_indicies_pages]

    if debug_print:
        print(f'page_grid_sizes: {page_grid_sizes}')
    return subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes
