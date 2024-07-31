import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray

import pyphoplacecellanalysis.External.pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView
from PyQt5 import QtCore
# from PyQt5.QtCore import QAbstractTableModel, Qt


# class pandasModel(QAbstractTableModel):
#     """ https://learndataanalysis.org/display-pandas-dataframe-with-pyqt5-qtableview-widget/ 
#     """

#     def __init__(self, data):
#         QAbstractTableModel.__init__(self)
#         self._data = data

#     def rowCount(self, parent=None):
#         return self._data.shape[0]

#     def columnCount(self, parnet=None):
#         return self._data.shape[1]

#     def data(self, index, role=Qt.DisplayRole):
#         if index.isValid():
#             if role == Qt.DisplayRole:
#                 return str(self._data.iloc[index.row(), index.column()])
#         return None

#     def headerData(self, col, orientation, role):
#         if orientation == Qt.Horizontal and role == Qt.DisplayRole:
#             return self._data.columns[col]
#         return None

# Newer: https://learndataanalysis.org/display-pandas-dataframe-with-pyqt5-qtableview-widget/



class PandasModel(QtCore.QAbstractTableModel):
    """ https://raw.githubusercontent.com/eyllanesc/stackoverflow/master/questions/44603119/PandasModel.py
    https://stackoverflow.com/questions/44603119/how-to-display-a-pandas-data-frame-with-pyqt5-pyside2
    
    
    """
    def __init__(self, df = pd.DataFrame(), parent=None): 
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()
        
        

# Step 1: Convert DataFrame to QAbstractTableModel
class SimplePandasModel(QtCore.QAbstractTableModel):
    """ 2023-12-13 Generated by ChatGPT 

    """
    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid() and role == QtCore.Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(self._data.index[section])
        return None




## 
# TypeError: Don't know how to iterate over data type: <class 'pandas.core.frame.DataFrame'>
# TypeError: Don't know how to iterate over data type: <class 'pyphocorehelpers.gui.Qt.pandas_model.SimplePandasModel'>

# ==================================================================================================================== #
# Helper functions                                                                                                     #
# ==================================================================================================================== #
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import _debug_plot_directional_template_rasters, build_selected_spikes_df, add_selected_spikes_df_points_to_scatter_plot

def create_tabbed_table_widget(dataframes_dict: Dict[str, pd.DataFrame]) -> Tuple[pg.QtWidgets.QTabWidget, Dict[str, SimplePandasModel], Dict[str, pg.QtWidgets.QTableView]]:
    """
    Creates a tabbed widget with three tables within the given layout.

    Args:
    ctrl_layout: The layout to add the tab widget to.
    dataframes: A list of three pandas.DataFrame objects to populate the tables.

    Returns:
    A QTabWidget containing the three tables.
    
    Usage:
        from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel, create_tabbed_table_widget

        ctrl_layout = pg.LayoutWidget()
        ctrl_widgets_dict = dict()
                                                                                          
        # Tabbled table widget:
        tab_widget, views_dict, models_dict = create_tabbed_table_widget(dataframes_dict={'epochs': active_epochs_df.copy(),
                                                                                          'spikes': global_spikes_df.copy(), 
                                                                                           'combined_epoch_stats': pd.DataFrame()})
        ctrl_widgets_dict['tables_tab_widget'] = tab_widget
        ctrl_widgets_dict['views_dict'] = views_dict
        ctrl_widgets_dict['models_dict'] = models_dict

        # Add the tab widget to the layout
        ctrl_layout.addWidget(tab_widget, row=2, rowspan=1, col=1, colspan=1)

    """

    # Create the tab widget and dictionaries
    tab_widget = pg.QtWidgets.QTabWidget()
    models_dict = {}
    views_dict = {}

    # Define tab names
    
    # Add tabs and corresponding views
    for i, (a_name, df) in enumerate(dataframes_dict.items()):
        # Create SimplePandasModel for each DataFrame
        models_dict[a_name] = SimplePandasModel(df.copy())

        # Create and associate view with model
        view = pg.QtWidgets.QTableView()
        view.setModel(models_dict[a_name])
        views_dict[a_name] = view

        # Add tab with view
        tab_widget.addTab(view, a_name)

    return tab_widget, views_dict, models_dict



# Step 2: Create PyQt Application
def build_test_app_containing_pandas_table(active_selected_spikes_df: pd.DataFrame):
    import pyphoplacecellanalysis.External.pyqtgraph as pg

    app = pg.mkQApp('test')
    # app = QApplication(sys.argv)

    # Step 3: Create main window and layout
    mainWindow = QMainWindow()
    centralWidget = pg.LayoutWidget()
    mainWindow.setCentralWidget(centralWidget)

    # Step 4: Create DataFrame and QTableView
    df =  active_selected_spikes_df # pd.DataFrame(...)  # Replace with your DataFrame
    # model = PandasModel(df)
    model = SimplePandasModel(df)
        
    tableView = QTableView()
    tableView.setModel(model)

    # Step 5: Add TableView to LayoutWidget
    centralWidget.addWidget(tableView)

    # Display the window
    mainWindow.show()
    # sys.exit(app.exec_())
        


# class pandasModel(QtCore.QAbstractTableModel):

#     def __init__(self, data):
#         QtCore.QAbstractTableModel.__init__(self)
#         self._data = data

#     def rowCount(self, parent=None):
#         return self._data.shape[0]

#     def columnCount(self, parnet=None):
#         return self._data.shape[1]

#     def data(self, index, role=Qt.DisplayRole):
#         if index.isValid():
#             if role == Qt.DisplayRole:
#                 return str(self._data.iloc[index.row(), index.column()])
#         return None

#     def headerData(self, col, orientation, role):
#         if orientation == Qt.Horizontal and role == Qt.DisplayRole:
#             return self._data.columns[col]
#         return None
    
    


# if __name__ == '__main__':
#     df = pd.DataFrame({'a': ['Mary', 'Jim', 'John'],
#                 'b': [100, 200, 300],
#                 'c': ['a', 'b', 'c']})
                   
#     app = QApplication(sys.argv)
#     model = pandasModel(df)
#     view = QTableView()
#     view.setModel(model)
#     view.resize(800, 600)
#     view.show()
#     sys.exit(app.exec_())


if __name__ == '__main__':
    sys.exit(app.exec_())
    # pg.exec()

