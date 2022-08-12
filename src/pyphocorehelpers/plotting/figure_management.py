import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt

class PhoActiveFigureManager2D(object):
    """Offers convenience methods for accessing and updating the extent (size and position on the screen) for the current Matplotlib figures (via its current_figure_manager property."""
    debug_print = False
    
    @property
    def current_figure_manager(self):
        """The current_figure_manager property."""
        return plt.get_current_fig_manager() # get the active figure manager
 
    @property
    def figure_nums(self):
        """The matplotlib figure numbers for active figures."""
        return plt.get_fignums()
    
    @property
    def figure_labels(self):
        """The matplotlib figure labels for active figures."""
        return plt.get_figlabels()
    
    @property
    def figures_dict(self):
        """The matplotlib figure labels for active figures."""
        return {fig_num:plt.figure(fig_num) for fig_num in self.figure_nums} # return a dictionary of the new figure_nums and figures
  
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
 
 
    ## Convinence methods:
    def close_all(self, debug_print=False):
        """ closes all open figures """
        for a_fig_id, a_fig in self.figures_dict.items():
            if debug_print:
                print(f'closing {a_fig_id}...')
            plt.close(a_fig_id) # this works to actually close the figure
            # a_fig.canvas.close() # this does not work to close the figure, the fig_man.figure_nums are still the same after calling
            # NOTE: there's also plt.close('all')
    
    
    @classmethod
    def get_geometries(cls, active_fig_mngr=None):
        if active_fig_mngr is None:
            active_fig_mngr = plt.get_current_fig_manager() # get the active figure manager
        # get the QTCore PyRect object
        geom = active_fig_mngr.window.geometry() # PyQt5.QtCore.QRect(8, 31, 5284, 834)
        x,y,dx,dy = geom.getRect() # (8, 31, 5284, 834)
        if cls.debug_print:
            print(f'geom: {geom}')
        extent = (x,y,dx,dy)
        return geom, extent

    @classmethod
    def set_geometries(cls, x:int=None, y:int=None, width:int=None, height:int=None):
        active_figure_man = plt.get_current_fig_manager() # get the active figure manager
        geom, prev_extent = cls.get_geometries(active_fig_mngr=active_figure_man)
        updated_extent = np.array(prev_extent)
        # extent is of the form (x,y,dx,dy)	
        if x is not None:
            updated_extent[0] = x
        if y is not None:
            updated_extent[1] = y
        if width is not None:
            updated_extent[2] = width
        if height is not None:
            updated_extent[3] = height
  
        newX, newY, newWidth, newHeight = updated_extent
        # and then set the new extents:
        active_figure_man.window.setGeometry(newX, newY, newWidth, newHeight)


    @classmethod
    def reshow_figure(cls, a_closed_figure):
        """ re-opens the window for a figure with a valid handle in memory (hasn't been garbage collected) but no active window (perhaps because it was previously closed with plt.close(the_fig) """
        # create a dummy figure and use its manager to display "a_closed_figure"
        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = a_closed_figure
        a_closed_figure.set_canvas(new_manager.canvas)
        return a_closed_figure


    """ Older Functions """
    @staticmethod
    def debug_print_matplotlib_figure_size(F):
        """ Prints the current figure size and DPI for a matplotlib figure F. 
        See https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib 
        Usage:
            SizeInches, DPI = debug_print_matplotlib_figure_size(a_fig)
        """
        DPI = F.get_dpi()
        print(f'DPI: {DPI}')
        SizeInches = F.get_size_inches()
        print(f'Default size in Inches: {SizeInches}')
        print('Which should result in a {} x {} Image'.format(DPI*SizeInches[0], DPI*SizeInches[1]))
        return SizeInches, DPI

    @staticmethod
    def rescale_figure_size(F, scale_multiplier=2.0, debug_print=False):
        """ Scales up the Matplotlib Figure by a factor of scale_multiplier (in both width and height) without distorting the fonts or line sizes. 
        Usage:
            rescale_figure_size(a_fig, scale_multiplier=2.0, debug_print=True)
        """
        CurrentSize = F.get_size_inches()
        F.set_size_inches((CurrentSize[0]*scale_multiplier, CurrentSize[1]*scale_multiplier))
        if debug_print:
            RescaledSize = F.get_size_inches()
            print(f'Size in Inches: {RescaledSize}')
        return F

    @staticmethod
    def panel_label(ax, label, fontsize=12):
        ax.text(
            x=-0.08,
            y=1.15,
            s=label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
            ha="right",
        )


def capture_new_figures_decorator(func):
    """ a decorator that's supposed to capture all matplotlib figures generated within the function that it decorates by comparing the global plt.get_fignums() before and after that function's executions. Won't detect updated figures, and hasn't been thoroughly tested. 
    
    """
    def wrapper(*args, **kwargs):
        pre_fignums = plt.get_fignums()
        print(f"pre_fignums: {pre_fignums}.")
        func(*args, **kwargs)
        post_fignums = plt.get_fignums()
        print(f"post_fignums: {post_fignums}.")
        new_fignums = list(set(post_fignums).symmetric_difference(set(pre_fignums)))
        print(f"new_fignums: {new_fignums}.")
        return {fig_num:plt.figure(fig_num) for fig_num in new_fignums} # return a dictionary of the new figure_nums and figures
    return wrapper



    
class FigureFormatter2D(object):
    """
    docstring for FigureFormatter2D.
    
    variant
    plot_type # the type of the plot
    
    
    
    title
    subtitle
    
    """
    def __init__(self, arg):
        super(FigureFormatter2D, self).__init__()
        self.arg = arg
      
    
      
    # def run(self):
    #     title_string = ' '.join([active_pf_1D_identifier_string])
    #     subtitle_string = ' '.join([f'{active_placefields1D.config.str_for_display(False)}'])
        
    #     plt.gcf().suptitle(title_string, fontsize='14')
    #     plt.gca().set_title(subtitle_string, fontsize='10') 
        
    # def save_to_disk(self):
    #     active_pf_1D_filename_prefix_string = f'Placefield1D-{active_epoch_name}'
    #     if variant_identifier_label is not None:
    #         active_pf_1D_filename_prefix_string = '-'.join([active_pf_1D_filename_prefix_string, variant_identifier_label])
    #     active_pf_1D_filename_prefix_string = f'{active_pf_1D_filename_prefix_string}-' # it always ends with a '-' character
    #     common_basename = active_placefields1D.str_for_filename(prefix_string=active_pf_1D_filename_prefix_string)
    #     active_pf_1D_output_filepath = active_config.plotting_config.get_figure_save_path(common_parent_foldername, common_basename).with_suffix('.png')
    #     print('Saving 1D Placefield image out to "{}"...'.format(active_pf_1D_output_filepath), end='')
    #     plt.savefig(active_pf_1D_output_filepath)
    #     print('\t done.')

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


    


def raise_window(figname=None):
    """ find the backend and use the appropriate method """
    def _raise_window_Qt(figname=None):
        """
        Raise the plot window for Figure figname to the foreground.  If no argument
        is given, raise the current figure.

        This function will only work with a Qt graphics backend.  It assumes you
        have already executed the command 'import matplotlib.pyplot as plt'.

        Usage:
            plt.figure('quartic')
            plt.plot(x, x**4 - x**2, 'b', lw=3)
            raise_window_Qt('quartic')

        """
        if figname: plt.figure(figname)
        cfm = plt.get_current_fig_manager()
        cfm.window.activateWindow()
        cfm.window.raise_()
        return cfm

    def _raise_window_Tk(figname=None):
        """
        Raise the plot window for Figure figname to the foreground.  If no argument
        is given, raise the current figure.

        This function will only work with a Tk graphics backend.  It assumes you
        have already executed the command 'import matplotlib.pyplot as plt'.

        Usage:
            plt.figure('quartic')
            plt.plot(x, x**4 - x**2, 'b', lw=3)
            raise_window_Tk('quartic')

        """

        if figname: plt.figure(figname)
        cfm = plt.get_current_fig_manager()
        cfm.window.attributes('-topmost', True)
        cfm.window.attributes('-topmost', False)
        return cfm

    # TODO: get current backend. Assumes Qt currently.
    backend_name = mpl.get_backend()
    if backend_name == 'TkAgg':
        return _raise_window_Tk(figname=figname)
    elif backend_name == 'WXAgg':
        raise NotImplementedError
    elif backend_name == 'Qt5Agg':
        return _raise_window_Qt(figname=figname)
    else:
        print(f"Unknown matplotlib backend type: {backend_name}")
        raise NotImplementedError
        
    
    

