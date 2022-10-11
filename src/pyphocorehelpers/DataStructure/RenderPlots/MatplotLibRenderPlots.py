from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots


class MatplotlibRenderPlots(RenderPlots):
	"""Container for holding and accessing Matplotlib-based figures for MatplotlibRenderPlots."""
	def __init__(self, name='MatplotlibRenderPlots', figures=[], axes=[], **kwargs):
		super(MatplotlibRenderPlots, self).__init__(name, figures = figures, axes=axes, **kwargs)
	

	
