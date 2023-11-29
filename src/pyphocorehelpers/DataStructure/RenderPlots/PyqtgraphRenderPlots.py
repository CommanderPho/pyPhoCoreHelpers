from typing import Dict
from pyphocorehelpers.DataStructure.general_parameter_containers import RenderPlots, RenderPlotsData, VisualizationParameters
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from attrs import define, field, Factory

class PyqtgraphRenderPlots(RenderPlots):
	"""Container for holding and accessing Pyqtgraph-based figures for PyqtgraphRenderPlots.

	from pyphocorehelpers.DataStructure.RenderPlots.PyqtgraphRenderPlots import PyqtgraphRenderPlots

	"""
	_display_library:str = 'pyqtgraph'
	
	def __init__(self, name='PyqtgraphRenderPlots', app=None, parent_root_widget=None, display_outputs=DynamicParameters(), context=None, **kwargs):
		super(PyqtgraphRenderPlots, self).__init__(name, app=app, parent_root_widget=parent_root_widget, display_outputs=display_outputs, context=context, **kwargs)
		



@metadata_attributes(short_name=None, tags=['unused', 'container', 'pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-17 20:06', related_items=[])
@define(slots=False)
class GenericPyQtGraphContainer:
    """ GenericPyQtGraphContainer holds related plots, their data, and methods that manipulate them in a straightforward way

    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import GenericPyQtGraphContainer

    """
    name: str = field(default='plot')
    params: VisualizationParameters = field(default=Factory(VisualizationParameters, 'plotter'))
    ui: PhoUIContainer = field(default=Factory(PhoUIContainer, 'plotter'))
    plots: PyqtgraphRenderPlots = field(default=Factory(PyqtgraphRenderPlots, 'plotter'))
    plot_data: RenderPlotsData = field(default=Factory(RenderPlotsData, 'plotter'))



@metadata_attributes(short_name=None, tags=['unused', 'container', 'pyqtgraph', 'interactive', 'scatterplot'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-17 20:06', related_items=[])
@define(slots=False)
class GenericPyQtGraphScatterClicker:
    """ GenericPyQtGraphContainer holds related plots, their data, and methods that manipulate them in a straightforward way

    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger import GenericPyQtGraphScatterClicker

    """
    lastClickedDict: Dict = field(default=Factory(dict))


    def on_scatter_plot_clicked(self, plot, evt):
        """ captures `lastClicked` 
        plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
        clicked points <MouseClickEvent (78.6115,-2.04825) button=1>

        """
        # global lastClicked  # Declare lastClicked as a global variable
        if plot not in self.lastClickedDict:
            self.lastClickedDict[plot] = None

        # for p in self.lastClicked:
        # 	p.resetPen()
        # print(f'plot: {plot}') # plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
        # print(f'\tevt: {evt}')	
        # print("clicked points", evt.pos()) # clicked points <MouseClickEvent (48.2713,1.32425) button=1>
        # print(f'args: {args}')
        pt_x, pt_y = evt.pos()
        idx_x = int(round(pt_x))
        print(f'\tidx_x: {idx_x}')
        # pts = plot.pointsAt(evt.pos())
        # print(f'pts: {pts}')
        # for p in points:
        # 	p.setPen(clickedPen)
        # self.lastClicked = idx_x
        self.lastClickedDict[plot] = idx_x




# lastClicked = []
# def _test_scatter_plot_clicked(plot, evt):
# 	""" captures `lastClicked` 
# 	plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
# 	clicked points <MouseClickEvent (78.6115,-2.04825) button=1>

# 	"""
# 	global lastClicked  # Declare lastClicked as a global variable
# 	# for p in lastClicked:
# 	# 	p.resetPen()
# 	# print(f'plot: {plot}') # plot: <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotDataItem.PlotDataItem object at 0x0000023C7D74C8B0>
# 	# print(f'\tevt: {evt}')	
# 	# print("clicked points", evt.pos()) # clicked points <MouseClickEvent (48.2713,1.32425) button=1>
# 	# print(f'args: {args}')
# 	pt_x, pt_y = evt.pos()
# 	idx_x = int(round(pt_x))
# 	print(f'\tidx_x: {idx_x}')
# 	# pts = plot.pointsAt(evt.pos())
# 	# print(f'pts: {pts}')
# 	# for p in points:
# 	# 	p.setPen(clickedPen)
# 	lastClicked = idx_x



