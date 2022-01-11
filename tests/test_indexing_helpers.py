import unittest
import numpy as np
import pandas as pd
# import the package
import sys, os
from pathlib import Path

# Add Neuropy to the path as needed
tests_folder = Path(os.path.dirname(__file__))

try:
    import pyphocorehelpers
except ModuleNotFoundError as e:    
    root_project_folder = tests_folder.parent
    print('root_project_folder: {}'.format(root_project_folder))
    src_folder = root_project_folder.joinpath('src')
    pyphocorehelpers_folder = src_folder.joinpath('pyphocorehelpers')
    print('pyphocorehelpers_folder: {}'.format(pyphocorehelpers_folder))
    sys.path.insert(0, str(src_folder))
finally:
    from pyphocorehelpers.indexing_helpers import get_bin_centers, build_spanning_bins, BinningInfo, compute_spanning_bins


class TestIndexingMethods(unittest.TestCase):

    def setUp(self):
        # Hardcoded:
        self.integer_bin_edges = np.array([0, 1, 2, 3, 4, 5])
        self.float_bin_edges = np.array([0.1, 1.1, 2.1, 3.1, 4.1, 5.1])
        
        # unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(sess.spikes_df.copy(), time_bin_size, debug_print=debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)

    def tearDown(self):
        self.integer_bin_edges=None
        self.float_bin_edges = None
        
        
    # def test_time_bin_spike_counts_N_i(self, out_digitized_variable_bins, out_binning_info):
    #     np.shape(out_digitized_variable_bins) # (85842,), array([  22.30206346,   22.32206362,   22.34206378, ..., 1739.09557005, 1739.11557021, 1739.13557036])
    #     assert out_digitized_variable_bins[-1] == out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!"
    #     assert out_digitized_variable_bins[0] == out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!"

    def test_get_bin_centers(self):
        bin_centers = get_bin_centers(self.integer_bin_edges)
        self.assertEqual((np.shape(self.integer_bin_edges)[0] - 1), np.shape(bin_centers)[0], 'bin_centers should be one element smaller than bin_edges')


    def test_compute_spanning_bins_num_bins_mode(self):
        fixed_num_bins = 32
        active_position_extents = np.array([23.92332935, 261.86436665])
        out_digitized_variable_bins, out_binning_info = compute_spanning_bins(active_position_extents, num_bins=fixed_num_bins)
        self.assertEqual(fixed_num_bins, out_binning_info.num_bins, f"out_binning_info.num_bins ({out_binning_info.num_bins}) must be equal to fixed_num_bins ({fixed_num_bins})! ")
        self.assertEqual(fixed_num_bins, np.shape(out_digitized_variable_bins)[0], f"np.shape(out_digitized_variable_bins)[0] ({np.shape(out_digitized_variable_bins)[0]}) must be equal to fixed_num_bins ({fixed_num_bins})! ")
        # Extents matching:
        self.assertEqual(out_digitized_variable_bins[-1], out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!")
        self.assertEqual(out_digitized_variable_bins[0], out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!")
        self.assertEqual(active_position_extents[-1], out_binning_info.variable_extents[1], "active_position_extents[-1] should be the maximum variable extent!")
        self.assertEqual(active_position_extents[0], out_binning_info.variable_extents[0], "active_position_extents[0] should be the minimum variable extent!")
        print(f'test_compute_spanning_bins_num_bins_mode(): out_digitized_variable_bins: {out_digitized_variable_bins}, out_binning_info: {out_binning_info}')
        
        
        # Y position:
        active_position_extents = np.array([87.58388756, 153.31348421])
        out_digitized_variable_bins, out_binning_info = compute_spanning_bins(active_position_extents, num_bins=fixed_num_bins)
        self.assertEqual(fixed_num_bins, out_binning_info.num_bins, f"out_binning_info.num_bins ({out_binning_info.num_bins}) must be equal to fixed_num_bins ({fixed_num_bins})! ")
        self.assertEqual(fixed_num_bins, np.shape(out_digitized_variable_bins)[0], f"np.shape(out_digitized_variable_bins)[0] ({np.shape(out_digitized_variable_bins)[0]}) must be equal to fixed_num_bins ({fixed_num_bins})! ")
        # Extents matching:
        self.assertEqual(out_digitized_variable_bins[-1], out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!")
        self.assertEqual(out_digitized_variable_bins[0], out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!")
        self.assertEqual(active_position_extents[-1], out_binning_info.variable_extents[1], "active_position_extents[-1] should be the maximum variable extent!")
        self.assertEqual(active_position_extents[0], out_binning_info.variable_extents[0], "active_position_extents[0] should be the minimum variable extent!")
        print(f'test_compute_spanning_bins_num_bins_mode(): out_digitized_variable_bins: {out_digitized_variable_bins}, out_binning_info: {out_binning_info}')


        
    def test_compute_spanning_bins_bin_size_mode(self):
        active_position_extents = np.array([23.92332935, 261.86436665])
        fixed_bin_size = 5
        out_digitized_variable_bins, out_binning_info = compute_spanning_bins(active_position_extents, bin_size=fixed_bin_size)
        print(f'test_compute_spanning_bins_bin_size_mode(): out_digitized_variable_bins: {out_digitized_variable_bins}, np.shape(out_digitized_variable_bins): {np.shape(out_digitized_variable_bins)}, out_binning_info: {out_binning_info}')
        self.assertEqual(out_binning_info.num_bins, np.shape(out_digitized_variable_bins)[0], f"out_binning_info.num_bins ({out_binning_info.num_bins}) must be equal to np.shape(out_digitized_variable_bins)[0] ({np.shape(out_digitized_variable_bins)[0]})!")
        # Extents matching:
        self.assertGreaterEqual(out_digitized_variable_bins[-1], out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!") # Note assertGreaterEqual for fixed_bin_size mode
        self.assertEqual(out_digitized_variable_bins[0], out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!")
        self.assertGreaterEqual(active_position_extents[-1], out_binning_info.variable_extents[1], "active_position_extents[-1] should be the maximum variable extent!") # Note assertGreaterEqual for fixed_bin_size mode
        self.assertEqual(active_position_extents[0], out_binning_info.variable_extents[0], "active_position_extents[0] should be the minimum variable extent!")
        
        # Y position:
        active_position_extents = np.array([87.58388756, 153.31348421])
        out_digitized_variable_bins, out_binning_info = compute_spanning_bins(active_position_extents, bin_size=fixed_bin_size)
        print(f'test_compute_spanning_bins_bin_size_mode(): out_digitized_variable_bins: {out_digitized_variable_bins}, np.shape(out_digitized_variable_bins): {np.shape(out_digitized_variable_bins)}, out_binning_info: {out_binning_info}')
        self.assertEqual(out_binning_info.num_bins, np.shape(out_digitized_variable_bins)[0], f"out_binning_info.num_bins ({out_binning_info.num_bins}) must be equal to np.shape(out_digitized_variable_bins)[0] ({np.shape(out_digitized_variable_bins)[0]})!")
        # Extents matching:
        self.assertGreaterEqual(out_digitized_variable_bins[-1], out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!") # Note assertGreaterEqual for fixed_bin_size mode
        self.assertEqual(out_digitized_variable_bins[0], out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!")
        self.assertGreaterEqual(active_position_extents[-1], out_binning_info.variable_extents[1], "active_position_extents[-1] should be the maximum variable extent!") # Note assertGreaterEqual for fixed_bin_size mode
        self.assertEqual(active_position_extents[0], out_binning_info.variable_extents[0], "active_position_extents[0] should be the minimum variable extent!")


    def test_build_spanning_bins_edges(self):
        # active_position_extents = np.array([23.92332935, 261.86436665, 87.58388756, 153.31348421])
        active_position_extents = np.array([23.92332935, 261.86436665])
        out_digitized_variable_bins, out_binning_info = build_spanning_bins(active_position_extents, max_bin_size=5)
        self.assertEqual(out_digitized_variable_bins[-1], out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!")
        self.assertEqual(out_digitized_variable_bins[0], out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!")
        self.assertEqual(active_position_extents[-1], out_binning_info.variable_extents[1], "active_position_extents[-1] should be the maximum variable extent!")
        self.assertEqual(active_position_extents[0], out_binning_info.variable_extents[0], "active_position_extents[0] should be the minimum variable extent!")
        
        # Y position:
        active_position_extents = np.array([87.58388756, 153.31348421])
        out_digitized_variable_bins, out_binning_info = build_spanning_bins(active_position_extents, max_bin_size=5)
        self.assertEqual(out_digitized_variable_bins[-1], out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!")
        self.assertEqual(out_digitized_variable_bins[0], out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!")
        self.assertEqual(active_position_extents[-1], out_binning_info.variable_extents[1], "active_position_extents[-1] should be the maximum variable extent!")
        self.assertEqual(active_position_extents[0], out_binning_info.variable_extents[0], "active_position_extents[0] should be the minimum variable extent!")

        
        # bin_centers = get_bin_centers(self.integer_bin_edges)
        # self.assertEqual((np.shape(self.integer_bin_edges)[0] - 1), np.shape(bin_centers)[0], 'bin_centers should be one element smaller than bin_edges')




if __name__ == '__main__':
    unittest.main()
    
    