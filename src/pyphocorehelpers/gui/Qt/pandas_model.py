import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import nptyping as ND
from nptyping import NDArray
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
import pyphoplacecellanalysis.External.pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QHeaderView, QStyleOptionHeader, QStyle
from PyQt5 import QtCore
# from PyQt5.QtCore import QAbstractTableModel, Qt

from PyQt5.QtCore import Qt, QRect, QSize, QModelIndex
from PyQt5.QtGui import QPainter, QPen
import pandas as pd

@metadata_attributes(short_name=None, tags=['widget', 'table', 'header', 'qt', 'gui'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-19 14:12', related_items=[])
class GroupedHeaderView(QHeaderView):
    """ tries to draw visually grouped table headers (so that each column can belong to a conceptual 'parent' category. 
    """
    def __init__(self, df, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.df = df
        self.groups = self._build_groups()
        self.setDefaultAlignment(Qt.AlignCenter)
        self.setStretchLastSection(True)

    def _build_groups(self):
        non_grouped_cols = ['start', 'stop', 'label', 'duration', 'is_user_annotated_epoch', 'is_valid_epoch', 'session_name', 'time_bin_size', 'delta_aligned_start_t', 'pre_post_delta_category', 'maze_id']
        all_cols = list(self.df.columns)
        start_idx = len(non_grouped_cols)
        grouped_cols = all_cols[start_idx:]
        group_map = {}
        for i, c in enumerate(grouped_cols, start=start_idx):
            prefix = c.rsplit('_', 2)[0]
            group_map.setdefault(prefix, []).append(i)

        groups = {}
        for i, c in enumerate(non_grouped_cols):
            groups[c] = [i]

        for prefix, indices in group_map.items():
            if len(indices) == 4:
                groups[prefix] = indices
            else:
                for i_idx in indices:
                    groups[all_cols[i_idx]] = [i_idx]

        return groups

    def sizeHint(self):
        # Increase the height for two-level headers
        base_hint = super().sizeHint()
        return QSize(base_hint.width(), base_hint.height()*2)

    def paintSection(self, painter, rect, logicalIndex):
        if not rect.isValid():
            return
        painter.save()
        # Draw background and border
        painter.fillRect(rect, self.palette().window())
        painter.drawRect(rect)

        # Draw column label at bottom half
        col_name = self.model().headerData(logicalIndex, Qt.Horizontal, Qt.DisplayRole) or ""
        half_height = rect.height() // 2
        bottom_rect = QRect(rect.x(), rect.y() + half_height, rect.width(), half_height)
        painter.drawText(bottom_rect, Qt.AlignCenter, col_name)
        painter.restore()

    def paintEvent(self, event):
        super().paintEvent(event)
        # Paint group labels on top half
        painter = QPainter(self.viewport())
        half_height = self.height()//2
        for group_name, cols in self.groups.items():
            if not cols:
                continue
            first_col = min(cols)
            width = sum(self.sectionSize(c) for c in cols)
            left = self.sectionViewportPosition(first_col)
            group_rect = QRect(left, 0, width, half_height)
            painter.drawText(group_rect, Qt.AlignCenter, group_name)


# Step 1: Convert DataFrame to QAbstractTableModel
@metadata_attributes(short_name=None, tags=['gui', 'pandas', 'dataframe', 'table', 'gui'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-19 14:12', related_items=[])
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
    
    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(self._data.index[section])
        return None


    def data(self, index, role=QtCore.Qt.DisplayRole):
        """ enables formatting the precision of the outputs, etc.
        """
        if not index.isValid() or role != QtCore.Qt.DisplayRole:
            return None
        val = self._data.iat[index.row(), index.column()]
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)


## 
# TypeError: Don't know how to iterate over data type: <class 'pandas.core.frame.DataFrame'>
# TypeError: Don't know how to iterate over data type: <class 'pyphocorehelpers.gui.Qt.pandas_model.SimplePandasModel'>

# ==================================================================================================================== #
# Helper functions                                                                                                     #
# ==================================================================================================================== #
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import _debug_plot_directional_template_rasters, build_selected_spikes_df, add_selected_spikes_df_points_to_scatter_plot

@function_attributes(short_name=None, tags=['tabbed', 'gui', 'table', 'USEFUL', 'gui', 'MAIN'], input_requires=[], output_provides=[], uses=['SimplePandasModel', 'GroupedHeaderView'], used_by=[], creation_date='2024-12-19 14:13', related_items=[])
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
        # view.setModel(df.to_numpy().__array_interface__) # Note: For a real model, use a QAbstractTableModel subclass. This is a placeholder.
        # header = GroupedHeaderView()
        header = GroupedHeaderView(df) ## needs the df now to determine header layout
        view.setHorizontalHeader(header)
        
        views_dict[a_name] = view

        # Add tab with view
        tab_widget.addTab(view, a_name)
        
        # Adjust the column widths to fit the contents
        view.resizeColumnsToContents()

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
    app = pg.mkQApp('vitables')
    sys.exit(app.exec_())
    # pg.exec()

