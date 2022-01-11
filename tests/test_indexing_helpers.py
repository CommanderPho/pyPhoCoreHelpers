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
    from pyphocorehelpers.indexing_helpers import get_bin_centers


class TestIndexingMethods(unittest.TestCase):

    def setUp(self):
        # Hardcoded:
        self.bin_edges = np.array([0, 1, 2, 3, 4, 5])
        # unit_specific_binned_spike_counts, out_digitized_variable_bins, out_binning_info = ZhangReconstructionImplementation.time_bin_spike_counts_N_i(sess.spikes_df.copy(), time_bin_size, debug_print=debug_print) # unit_specific_binned_spike_counts.to_numpy(): (40, 85841)

    def tearDown(self):
        self.bin_edges=None
        
    # def test_time_bin_spike_counts_N_i(self, out_digitized_variable_bins, out_binning_info):
    #     np.shape(out_digitized_variable_bins) # (85842,), array([  22.30206346,   22.32206362,   22.34206378, ..., 1739.09557005, 1739.11557021, 1739.13557036])
    #     assert out_digitized_variable_bins[-1] == out_binning_info.variable_extents[1], "out_digitized_variable_bins[-1] should be the maximum variable extent!"
    #     assert out_digitized_variable_bins[0] == out_binning_info.variable_extents[0], "out_digitized_variable_bins[0] should be the minimum variable extent!"

    def test_get_bin_centers(self):
        bin_centers = get_bin_centers(self.bin_edges)
        self.assertEqual((np.shape(self.bin_edges)[0] - 1), np.shape(bin_centers)[0], 'bin_centers should be one element smaller than bin_edges')


if __name__ == '__main__':
    unittest.main()
    
    