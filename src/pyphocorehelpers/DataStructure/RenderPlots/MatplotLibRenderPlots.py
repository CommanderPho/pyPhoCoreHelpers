from typing import Optional
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, FigureBase # FigureBase: both Figure and SubFigure
from matplotlib.axes import Axes

from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes


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
    def init_from_any_objects(cls, *args, name: Optional[str] = None, **kwargs):
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
        return cls(name=name, figures=figures, axes=axes)


    @classmethod
    def init_subplots(cls, *args, **kwargs):
        """ wraps the matplotlib.plt.subplots(...) command to initialize a new object with custom subplots. 
        2023-05-30 - Should this be here ideally? Is there a better class? I remember a MATPLOTLIB
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(*args, **kwargs)
        # check for scalar axes and wrap it in a tuple if needed before setting self.
        return cls(figures=[fig], axes=axes)



class FigureCollector:
    """ 
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    
    
    """
    def __init__(self, name='MatplotlibRenderPlots', figures=None, axes=None, contexts=None, base_context=None):
        self.name = name
        self.figures = figures or []
        self.axes = axes or []
        self.contexts = contexts or []
        self.base_context = base_context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup code, if needed
        pass

    def create_figure(self, *args, **kwargs):
        fig = plt.figure(*args, **kwargs)
        self.figures.append(fig)
        return fig
    
    def subplots(self, *args, **kwargs):
        fig, axes = plt.subplots(*args, **kwargs) # tuple[Figure, np.ndarray] or tuple[Figure, Axes]
        self.figures.append(fig)
        if isinstance(axes, Axes):
            self.axes.append(axes) # single scalar axis
        else:
            for ax in axes:
                self.axes.append(ax)
        return fig, axes
    


# #TODO 2023-12-23 22:01: - [ ] Context-determining figure
# class ContextCollectingFigureCollector(FigureCollector):
#     """ 
#     from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    
    
#     """
#     def __init__(self, name='ContextCollectingFigureCollector', figures=None, axes=None, context=None, context_list=None):
#         ## TODO: store specific properties:
#         self.context_list = context_list or []
#         super().__init__(name=name, figures=figures, axes=axes, context=context)
        

#     def create_figure(self, *args, **kwargs):
#         fig = plt.figure(*args, **kwargs)
#         self.figures.append(fig)
#         return fig
    
#     def subplots(self, *args, **kwargs):
#         sub_name = kwargs.pop('name', None) # 'name' is not a valid argument to plt.subplots anyway.
#         if sub_name is None:
#             # try to get subname from one of the other parameters:
#             num_name = kwargs.get('num', None)
#             sub_name = num_name or ""
               
#         if sub_name is not None:
#             if self.context is not None:
#                 # self.context.
#                 sub_context = self.context.
#             else:
#                 sub_context = IdentifyingContext(sub_name=sub_name)
#                 sub_context.adding_context_if_missing(sub_name=sub_name)
                                
#         fig, axes = plt.subplots(*args, **kwargs) # tuple[Figure, np.ndarray] or tuple[Figure, Axes]
#         self.figures.append(fig)
#         if isinstance(axes, Axes):
#             self.axes.append(axes) # single scalar axis
#         else:
#             for ax in axes:
#                 self.axes.append(ax)
#         return fig, axes
    



#TODO 2023-12-13 00:57: - [ ] Custom Figure (TODO)

@metadata_attributes(short_name=None, tags=['not-yet-implemented', 'matplotlib', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-13 00:56', related_items=[])
class MatplotlibRenderPlotsFigure(Figure):
    """A figure with emedded data/functions/etc.
    
    
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlotsFigure
    
    """

    def __init__(self, *args, name='MatplotlibRenderPlots', figures=[], axes=[], context=None, watermark=None, **kwargs):
        super().__init__(name, *args, figures=figures, axes=axes, context=context, **kwargs) # is this right or do I set them somewhere else?

        # if watermark is not None:
        #     bbox = dict(boxstyle='square', lw=3, ec='gray',
        #                 fc=(0.9, 0.9, .9, .5), alpha=0.5)
        #     self.text(0.5, 0.5, watermark,
        #               ha='center', va='center', rotation=30,
        #               fontsize=40, color='gray', alpha=0.5, bbox=bbox)



"""
from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlotsFigure

x = np.linspace(-3, 3, 201)
y = np.tanh(x) + 0.1 * np.cos(5 * x)

plt.figure(FigureClass=MatplotlibRenderPlotsFigure, watermark='draft')
plt.plot(x, y)
"""

