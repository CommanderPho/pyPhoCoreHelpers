import sys
import traceback
import types
import os
from typing import Optional
from functools import wraps

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir


def pyqtExceptionPrintingSlot(*args):
	""" replacement for @QtCore.Slot(...) that enables printing exceptions intead of failing silently!
	Very useful!

	from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
    @pyqtExceptionPrintingSlot(object)
    def on_tree_item_double_clicked(self, item, column):
        print(f"Item double-clicked: {item}, column: {column}\n\t", item.text(column))
        # print(f'\titem.data: {item.data}')
        # raise NotImplementedError
        item_data = item.data(column, 0) # ItemDataRole 
        print(f'\titem_data: {item_data}')
        a_fn_handle = self.curr_active_pipeline.plot.__getattr__(item_data)
        return a_fn_handle()


	"""
    if len(args) == 0 or isinstance(args[0], types.FunctionType):
        args = []
    @QtCore.pyqtSlot(*args)
    def slotdecorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args)
            except:
                print("Uncaught Exception in slot")
                traceback.print_exc()
        return wrapper

    return slotdecorator