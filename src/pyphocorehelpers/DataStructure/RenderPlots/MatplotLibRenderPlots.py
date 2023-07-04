import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, FigureBase # FigureBase: both Figure and SubFigure
from matplotlib.axes import Axes

from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots


class MatplotlibRenderPlots(RenderPlots):
    """Container for holding and accessing Matplotlib-based figures for MatplotlibRenderPlots.
    Usage:
        from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
    
    2023-05-30 - Updated to replace subplots.
    
    """
    _display_library:str = 'matplotlib'
    
    def __init__(self, name='MatplotlibRenderPlots', figures=[], axes=[], context=None, **kwargs):
        super(MatplotlibRenderPlots, self).__init__(name, figures = figures, axes=axes, context=context, **kwargs)
        

    @classmethod
    def init_from_any_objects(cls, *args):
        """ initializes a MatplotlibRenderPlots instance from any list containing Matplotlib objects (figures, axes, artists, contexts, etc).
        
        For the most base functionality we really only need the figures and axes. 

        """
        figures = []
        axes = [] # would make way more sense if we had a list of axes for each figure, and a list of Artists for each axes, etc. But this is a start.
        
        for obj in args:
            if isinstance(obj, FigureBase): # .Figure or .SubFigure
                figures.append(obj)
            elif isinstance(obj, Axes):
                axes.append(obj)
            # Add more elif blocks for other types if needed
        return cls(figures=figures, axes=axes)


    @classmethod
    def init_subplots(cls, *args, **kwargs):
        """ wraps the matplotlib.plt.subplots(...) command to initialize a new object with custom subplots. 
        2023-05-30 - Should this be here ideally? Is there a better class? I remember a MATPLOTLIB
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(*args, **kwargs)
        # check for scalar axes and wrap it in a tuple if needed before setting self.
        return cls(figures=[fig], axes=axes)
