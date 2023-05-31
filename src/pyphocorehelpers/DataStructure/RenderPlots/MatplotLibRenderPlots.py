from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots


class MatplotlibRenderPlots(RenderPlots):
	"""Container for holding and accessing Matplotlib-based figures for MatplotlibRenderPlots.
	Usage:
		from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
	
	2023-05-30 - Updated to replace subplots.
	
	"""
	def __init__(self, name='MatplotlibRenderPlots', figures=[], axes=[], **kwargs):
		super(MatplotlibRenderPlots, self).__init__(name, figures = figures, axes=axes, **kwargs)
		
	@classmethod
	def init_subplots(cls, *args, **kwargs):
		""" wraps the matplotlib.plt.subplots(...) command to initialize a new object with custom subplots. 
		2023-05-30 - Should this be here ideally? Is there a better class? I remember a MATPLOTLIB
		"""
		import matplotlib.pyplot as plt
		fig, axes = plt.subplots(*args, **kwargs)
		# check for scalar axes and wrap it in a tuple if needed before setting self.
		return cls(figures=[fig], axes=axes)
