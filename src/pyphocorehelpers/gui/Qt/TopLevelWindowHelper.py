from indexed import IndexedOrderedDict


class TopLevelWindowHelper:
    """ Not quite finished, but tools to interact with the active QtWidgets and QWindows at the top-level
    
    # app = pg.mkQApp()
    app = pg.mkQApp(spike_raster_window.applicationName) # <PyQt5.QtWidgets.QApplication at 0x1d44a4891f0>
    
    TopLevelWindowHelper.top_level_windows(app)
    """
    @classmethod
    def all_widgets(cls, app, searchType=None):
        """
            searchType: Type - e.g. SpikeRasterBase
        """
        # All widgets
        all_widgets_list = app.allWidgets()
        if searchType is not None:
            # Only widgets that inherit from searchType
            all_widgets_with_superclass_list = [a_widget for a_widget in all_widgets_list if isinstance(a_widget, (SpikeRasterBase))]
            return all_widgets_with_superclass_list
        else:
            # return all widgets
            return all_widgets_list
        
    @classmethod
    def top_level_windows(cls, app, only_visible=True):
        # All windows: returns a list of QWindow objects, may be more than are active:
        top_level_windows = app.allWindows()
        # top_level_window_names = [a_window.objectName() for a_window in top_level_windows] #['Spike3DRaster_VedoClassWindow', 'QFrameClassWindow', 'QWidgetClassWindow', 'PhoBaseMainWindowClassWindow']
        # top_level_window_is_visible = [a_window.isVisible() for a_window in top_level_windows]
        # top_level_window_names #['Spike3DRaster_VedoClassWindow', 'QFrameClassWindow', 'QWidgetClassWindow', 'PhoBaseMainWindowClassWindow']
        # top_level_window_is_visible
        # returns a dictionary of window_name:windows
        if only_visible:
            return IndexedOrderedDict({a_window.objectName():a_window for a_window in top_level_windows if a_window.isVisible()})
        else:
            return IndexedOrderedDict({a_window.objectName():a_window for a_window in top_level_windows})
        
        
    @classmethod
    def find_all_children_of_type(cls, app, searchType):
        """
        searchType: Type - e.g. Spike3DRaster
        """
        children = app.findChildren(searchType)
        return children
