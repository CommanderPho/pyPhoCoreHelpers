from qtpy import QtCore, QtWidgets, uic


# @function_attributes(short_name=None, tags=['UNUSED', 'UNTESTED', 'alternative', 'post-hoc', 'qt-creator', 'pyqt5', 'spacers', 'workaround'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 03:45', related_items=['load_ui_with_named_spacers'])
# def attach_named_spacers(ui_object, layout):
#     """Alternative to `load_ui_with_named_spacers` that uses the normal uic.load(...) and then finds the spacers post-hoc in the layout and attaches them to the ui object by name.
    
#     """
#     for i in range(layout.count()):
#         layout_item = layout.itemAt(i)
#         if layout_item.spacerItem():
#             # Get the name from the object name property
#             spacer_name = layout_item.spacerItem().objectName()
#             if spacer_name:
#                 setattr(ui_object, spacer_name, layout_item.spacerItem())


# @function_attributes(short_name=None, tags=['uic', 'qt-creator', 'pyqt5', 'spacers', 'workaround'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 03:45', related_items=[])
def load_ui_with_named_spacers(ui_file, base_instance=None):
    """Loads a UI file and makes spacers accessible by name.
    
    from pyphocorehelpers.gui.Qt.QtUIC_Helpers import load_ui_with_named_spacers
    
    Replaces `uic.load(...)` initialization:
    
    ```
    class Spike3DRasterLeftSidebarControlBar(QWidget):
        def __init__(self, parent=None):
			super().__init__(parent=parent) # Call the inherited classes __init__ method
			self.ui = uic.loadUi(uiFile, self) # Load the .ui file
			self.initUI()
			self.show() # Show the GUI
    ```
    with
    ```
    class Spike3DRasterLeftSidebarControlBar(QWidget):
        def __init__(self, parent=None):
			super().__init__(parent=parent) # Call the inherited classes __init__ method
			self.ui = load_ui_with_spacers(uiFile, self) # Load the .ui file
			self.initUI()
			self.show() # Show the GUI
	```
    
    """
    if base_instance is None:
        ui = uic.loadUi(ui_file)
    else:
        ui = uic.loadUi(ui_file, base_instance)
    
    # Process all layouts in the UI to find spacers
    def process_layout(layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.spacerItem() and hasattr(item.spacerItem(), 'objectName'):
                name = item.spacerItem().objectName()
                if name:
                    setattr(ui, name, item.spacerItem())
            elif item.layout():
                process_layout(item.layout())
            elif item.widget() and item.widget().layout():
                process_layout(item.widget().layout())
    
    # Start with the main layout
    if hasattr(ui, 'layout'):
        process_layout(ui.layout())
    
    return ui

