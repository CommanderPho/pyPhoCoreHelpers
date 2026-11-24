"""Tests for MATLAB-style matplotlib figure serialization (``.mplfig``)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

tests_folder = Path(os.path.dirname(__file__))

try:
    import pyphocorehelpers
except ModuleNotFoundError:
    root_project_folder = tests_folder.parent
    src_folder = root_project_folder.joinpath('src')
    sys.path.insert(0, str(src_folder))
finally:
    from pyphocorehelpers.plotting.figure_serialization import MplFigArchive, load_mplfig, save_mplfig


class TestFigureSerialization(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)


    def tearDown(self):
        plt.close('all')
        self._tmpdir.cleanup()


    def _make_line_figure(self):
        x = np.linspace(0.0, 1.0, 11)
        y = np.sin(2.0 * np.pi * x)
        fig, ax = plt.subplots()
        ax.plot(x, y, label='sine')
        ax.set_title('sine wave')
        fig.suptitle('test_mplfig_suptitle')
        return fig, ax, x, y


    def test_round_trip_preserves_axes_line_data_and_titles(self):
        fig, ax, x, y = self._make_line_figure()
        out_path = self.tmp_path / 'round_trip.mplfig'
        saved = save_mplfig(fig, out_path)
        self.assertEqual(saved, out_path)
        self.assertTrue(saved.exists())

        plt.close('all')
        archive = load_mplfig(saved, attach_to_current_pyplot=True)
        self.assertIsInstance(archive, MplFigArchive)
        self.assertIs(archive.fig, archive.figure)
        self.assertEqual(len(archive.figure.axes), 1)

        loaded_ax = archive.figure.axes[0]
        lines = loaded_ax.get_lines()
        self.assertGreaterEqual(len(lines), 1)
        np.testing.assert_allclose(lines[0].get_ydata(), y)
        np.testing.assert_allclose(lines[0].get_xdata(), x)
        self.assertEqual(loaded_ax.get_title(), 'sine wave')
        self.assertIsNotNone(archive.figure._suptitle)
        self.assertEqual(archive.figure._suptitle.get_text(), 'test_mplfig_suptitle')


    def test_round_trip_with_plot_data(self):
        fig, ax, x, y = self._make_line_figure()
        plot_data = {'x': x.copy(), 'y': y.copy(), 'note': 'sidecar'}
        metadata = {'session': 'test_session'}
        out_path = self.tmp_path / 'with_data'
        saved = save_mplfig(fig, out_path, plot_data=plot_data, metadata=metadata)
        self.assertEqual(saved.suffix, '.mplfig')

        plt.close('all')
        archive = load_mplfig(saved, attach_to_current_pyplot=False)
        self.assertIn('x', archive.plot_data)
        self.assertIn('y', archive.plot_data)
        np.testing.assert_allclose(archive.plot_data['x'], x)
        np.testing.assert_allclose(archive.plot_data['y'], y)
        self.assertEqual(archive.plot_data['note'], 'sidecar')
        self.assertEqual(archive.metadata.get('session'), 'test_session')
        self.assertEqual(archive.path, saved)


    def test_overwrite_false_raises_when_exists(self):
        fig, ax, x, y = self._make_line_figure()
        out_path = self.tmp_path / 'no_overwrite.mplfig'
        save_mplfig(fig, out_path, overwrite=True)
        with self.assertRaises(FileExistsError):
            save_mplfig(fig, out_path, overwrite=False)


if __name__ == '__main__':
    unittest.main()
