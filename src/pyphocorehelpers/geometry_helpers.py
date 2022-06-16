import numpy as np

from pyphocorehelpers.DataStructure.data_structure_builders import Width_Height_Tuple, cartesian_product

## Centroid point for camera
def centeroidnp(arr):
    """ Calculate the centroid of an array of points
        History: refactored from `pyphoplacecellanalysis.PhoPositionalData.analysis.helpers`
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def min_max_bounds(arr):
    """ Calculate the min and max of an array of points
        History: refactored from `pyphoplacecellanalysis.PhoPositionalData.analysis.helpers`
    """
    max_x = np.amax(arr[:, 0])
    max_y = np.amax(arr[:, 1])
    min_x = np.amin(arr[:, 0])
    min_y = np.amin(arr[:, 1])
    return [min_x, max_x, min_y, max_y]


def bounds_midpoint(arr):
    """ calculates the (x, y) midpoint given input in the format [min_x, max_x, min_y, max_y]
        History: refactored from `pyphoplacecellanalysis.PhoPositionalData.analysis.helpers`
    """
    min_x, max_x, min_y, max_y = arr
    return [(min_x + max_x)/2.0, (min_y + max_y)/2.0]


## General data tools:
def min_max_1d(data):
    return np.nanmin(data), np.nanmax(data)
    
def compute_data_extent(xpoints, *other_1d_series):
    """Computes the outer bounds, or "extent" of one or more 1D data series.

    Args:
        xpoints ([type]): [description]
        other_1d_series: any number of other 1d data series

    Returns:
        xmin, xmax, ymin, ymax, imin, imax, ...: a flat list of paired min, max values for each data series provided.
        
    Usage:
        # arbitrary number of data sequences:        
        xmin, xmax, ymin, ymax, x_center_min, x_center_max, y_center_min, y_center_max = compute_data_extent(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers)
        print(xmin, xmax, ymin, ymax, x_center_min, x_center_max, y_center_min, y_center_max)

        # simple 2D extent:
        extent = compute_data_extent(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin)
        print(extent)
    """
    num_total_series = len(other_1d_series) + 1 # + 1 for the x-series
    # pre-allocate output:     
    extent = np.empty(((2 * num_total_series),))
    # Do first-required series:
    xmin, xmax = min_max_1d(xpoints)
    extent[0], extent[1] = [xmin, xmax]
    # finish remaining series passed as inputs.
    for (i, a_series) in enumerate(other_1d_series):
        curr_min, curr_xmax = min_max_1d(a_series)
        curr_start_idx = 2 * (i + 1)
        extent[curr_start_idx] = curr_min
        extent[curr_start_idx+1] = curr_xmax
    return extent

def corner_points_from_extents(extents):
    """  Gets the corner points of the bounding shape specified by extents.
    Usage:
        xmin=23.923329354140844, xmax=263.92332935414083, ymin=123.85967782096927, ymax=153.85967782096927
        points = corner_points_from_extents([xmin, xmax, ymin, ymax])
        extent_pairs_list: [[23.923329354140844, 263.92332935414083], [123.85967782096927, 153.85967782096927]]

        points: array([[ 23.92332935, 123.85967782],
               [ 23.92332935, 153.85967782],
               [263.92332935, 123.85967782],
               [263.92332935, 153.85967782]])

    Example 2:
        points = corner_points_from_extents([23, 260, 124, 154])
            extent_pairs_list: [[23, 260], [124, 154]]
            points: [[ 23 124]
             [ 23 154]
             [260 124]
             [260 154]]

    Example 3:
        points = corner_points_from_extents([23, 260, 124, 154, 0, 1])
            extent_pairs_list: [[23, 260], [124, 154], [0, 1]]
            points: [[ 23 124   0]
             [ 23 124   1]
             [ 23 154   0]
             [ 23 154   1]
             [260 124   0]
             [260 124   1]
             [260 154   0]
             [260 154   1]]
    """
    num_extents = len(extents)
    dim_data = int(num_extents / 2)
    
    extent_pairs_list = [list([extents[2*i], extents[(2*i) + 1]]) for i in np.arange(dim_data)]
    print(f'extent_pairs_list: {extent_pairs_list}')
    points = cartesian_product(extent_pairs_list)
    # TODO: sort by z, y, x
    # np.lexsort(
    # points = np.sort(points, axis=2)
    return points
 
def compute_data_aspect_ratio(xbin, ybin, sorted_inputs=True):
    """Computes the aspect ratio of the provided data

    Args:
        xbin ([type]): [description]
        ybin ([type]): [description]
        sorted_inputs (bool, optional): whether the input arrays are pre-sorted in ascending order or not. Defaults to True.

    Returns:
        float: The aspect ratio of the data such that multiplying any height by the returned float would result in a width in the same aspect ratio as the data.
    """
    if sorted_inputs:
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1]) # assumes-pre-sourced events, which is valid for bins but not general
    else:
        xmin, xmax, ymin, ymax = compute_data_extent(xbin, ybin) # more general form.

    # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
    # extent = (xmin, xmax, ymin, ymax)
    
    width = xmax - xmin
    height = ymax - ymin
    
    aspect_ratio = width / height
    return aspect_ratio, Width_Height_Tuple(width, height)
    


def find_ranges_in_window(epoch_starts, epoch_ends, active_window):
    """ finds the epochs that lie either wholly or partially within the active_Window.
    This can be used for example to find the epochs that fall within the visualization window to know which rectangles to draw.
    
    Inputs:
        epoch_starts: an array of size {N,}
        epoch_ends: an array of size {N,}
        active_window: a tuple or list containing 2 elements representing the (win_start, win_end) times of the current window
        
    Returns:
        is_range_in_window: a boolean array of size {N,} that indicates whether the epoch is (either partially or fully) within the active window.
        included_epoch_indicies: an array of size {F,} containing the index of the original input epochs
        included_epoch_starts: an array of size {F,} containing the (potentially modified if truncated) start times for each epoch in the window
        included_epoch_ends: an array of size {F,} containing the (potentially modified if truncated) stop times for each epoch in the window
        included_epoch_is_truncated: an array of size {F,} containing a boolean value indicating whether that epoch was truncated by the edges of the window

    Usage:
        is_range_in_window, included_epoch_indicies, included_epoch_starts, included_epoch_ends, included_epoch_is_truncated = find_ranges_in_window(curr_sess.pbe.starts, curr_sess.pbe.stops, (598.65, 820.0))
        print(f'included_epoch_indicies: {np.shape(included_epoch_indicies)}')
        print(f'included_epoch_starts: {np.shape(included_epoch_starts)}')
        # build a dataframe to preview output
        pd.DataFrame({'included_epoch_indicies':included_epoch_indicies, 'included_epoch_starts':included_epoch_starts, 'included_epoch_ends':included_epoch_ends, 'included_epoch_is_truncated': included_epoch_is_truncated})
    
    
    """
    active_window_start, active_window_end = active_window
    # found_start_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,0], side='left')
    # found_end_indicies = np.searchsorted(times_arr, start_stop_times_arr[:,1], side='right') # find the end of the range
    
    # find all epochs that end before the start of the window to exclude them.
    # found_excluded_early_windows = (epoch_ends >= active_window_start)
        
    # find all epochs that start after the end of the window to exclude them.
    # found_excluded_late_windows = (epoch_starts >= active_window_end)
    
    # Combined for efficiency
    excluded_windows = np.logical_or((epoch_ends <= active_window_start), (epoch_starts >= active_window_end))
    is_range_in_window = np.logical_not(excluded_windows) # if True, range is at least partially in the window
    
    included_epoch_indicies = np.squeeze(np.where(is_range_in_window)) # np.where returns (1, 39) for some reason instead of (39,)
    included_epoch_starts = epoch_starts[included_epoch_indicies]
    included_epoch_ends = epoch_ends[included_epoch_indicies]
    included_epoch_is_truncated = np.full_like(included_epoch_starts, False)
    
    # TODO: return if empty
    if len(included_epoch_starts) > 0:
        # Left truncation occurs when (included_epoch_starts[0] < active_window_start)
        if (included_epoch_starts[0] < active_window_start):
            # left truncation
            included_epoch_is_truncated[0] = True
            included_epoch_starts[0] = active_window_start

        # Right truncation occurs when (included_epoch_ends[-1] > active_window_end)
        if (included_epoch_ends[-1] > active_window_end):
            # right truncation
            included_epoch_is_truncated[-1] = True
            included_epoch_starts[-1] = active_window_end
            
    return is_range_in_window, included_epoch_indicies, included_epoch_starts, included_epoch_ends, included_epoch_is_truncated
    
    
    