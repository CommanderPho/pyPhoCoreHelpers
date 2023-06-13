from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

class PyqtgraphRenderPlots(RenderPlots):
	"""Container for holding and accessing Pyqtgraph-based figures for PyqtgraphRenderPlots."""
	_display_library:str = 'pyqtgraph'
	
	def __init__(self, name='PyqtgraphRenderPlots', app=None, parent_root_widget=None, display_outputs=DynamicParameters(), context=None, **kwargs):
		super(PyqtgraphRenderPlots, self).__init__(name, app=app, parent_root_widget=parent_root_widget, display_outputs=display_outputs, context=context, **kwargs)
		
