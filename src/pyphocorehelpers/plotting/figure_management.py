import matplotlib.pyplot as plt

class PhoActiveFigureManager2D(object):
    """Offers convenience methods for accessing and updating the extent (size and position on the screen) for the current Matplotlib figures (via its current_figure_manager property."""
    
    @property
    def current_figure_manager(self):
        """The current_figure_manager property."""
        return plt.get_current_fig_manager() # get the active figure manager

    @property
    def window_extent(self):
        """The window_extent property."""
        return plt.get_current_fig_manager() # get the active figure manager
 
 
    @property
    def window_extent(self):
        """The window_extent property."""
        return PhoActiveFigureManager2D.get_geometries()
    @window_extent.setter
    def window_extent(self, value):
        PhoActiveFigureManager2D.set_geometries(*value)

 
    def __init__(self, name=''):
        super(PhoActiveFigureManager2D, self).__init__()
        self.name = name
    
    
    @classmethod
    def get_geometries(cls, active_fig_mngr=None):
        if active_fig_mngr is None:
            active_fig_mngr = plt.get_current_fig_manager() # get the active figure manager
        # get the QTCore PyRect object
        geom = active_fig_mngr.window.geometry() # PyQt5.QtCore.QRect(8, 31, 5284, 834)
        x,y,dx,dy = geom.getRect() # (8, 31, 5284, 834)
        print(f'geom: {geom}')
        extent = (x,y,dx,dy)
        return geom, extent

    @classmethod
    def set_geometries(cls, x:int=None, y:int=None, width:int=None, height:int=None):
        active_figure_man = plt.get_current_fig_manager() # get the active figure manager
        geom, prev_extent = cls.get_geometries(active_fig_mngr=active_figure_man)
        updated_extent = prev_extent.copy()
        # extent is of the form (x,y,dx,dy)	
        if x is not None:
            updated_extent[0] = x
        if y is not None:
            updated_extent[1] = y
        if width is not None:
            updated_extent[3] = width
        if height is not None:
            updated_extent[4] = height
  
        newX, newY, newWidth, newHeight = updated_extent
        # and then set the new extents:
        active_figure_man.window.setGeometry(newX, newY, newWidth, newHeight)


""" Further matplot exploration/prototyping

# active_fig_mngr = plt.get_current_fig_manager()
# active_fig = plt.gcf()
# active_ax = plt.gca()

# active_fig_mngr.canvas
# active_fig.get_window_extent()
# active_fig.subplotpars.left
# active_fig.get_children()
# active_fig.get_axes()
# active_fig_mngr.window.

# top=1.0,
# bottom=0.0,
# left=0.0,
# right=1.0,
# hspace=0.2,
# wspace=0.2

# plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)




"""