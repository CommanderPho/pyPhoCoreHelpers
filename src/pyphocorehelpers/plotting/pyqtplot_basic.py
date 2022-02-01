# required to enable non-blocking interaction:
# from PyQt5.Qt import QApplication
# # start qt event loop
# _instance = QApplication.instance()
# if not _instance:
#     _instance = QApplication([])
# app = _instance
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from dataclasses import dataclass

@dataclass
class BasicPyQtPlotApp(object):
    """Docstring for BasicPyQtPlotApp."""
    app: QtGui.QApplication
    win: QtGui.QMainWindow
    # w: pg.GraphicsLayoutWidget


def pyqtplot_common_setup(a_title):
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')
    pg.setConfigOptions(antialias = True)
    app = pg.mkQApp(a_title)
    # print(f'type(app): {type(app)}')
    # Create window to hold the image:
    win = QtGui.QMainWindow()
    win.resize(1600, 1600)
    # Creating a GraphicsLayoutWidget as the central widget
    w = pg.GraphicsLayoutWidget()
    win.setCentralWidget(w)
    
    return w, win, app