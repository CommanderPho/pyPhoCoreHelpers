from typing import Optional, List, Dict
import ipywidgets as widgets
from IPython.display import display



def build_fn_bound_buttons(button_defns, **default_kwargs):
	""" much simplier version of `JupyterButtonRowWidget` """
	# default_kwargs = dict(display='flex', flex_flow='column', align_items='stretch', layout=btn_layout)
	# button_list = []
	button_dict = {}
	for (a_label, a_fn, *args) in button_defns:
		if len(args) > 0:
			button_kwargs = args[0]
			# print(f'using provided button_kwargs: {button_kwargs}')
		else:
			button_kwargs = default_kwargs
			
		a_btn = widgets.Button(description=a_label, **button_kwargs) # , style= {'width': 'initial'}
		a_btn.on_click(a_fn)
		# button_list.append(a_btn)
		button_dict[a_label] = a_btn
	# return button_list
	return button_dict
		
		
class JupyterButtonRowWidget:
	""" Displays a clickable row of buttons in the Jupyter Notebook that perform any function 
	
	Usage:
		from pyphocorehelpers.gui.Jupyter.JupyterButtonRowWidget import JupyterButtonRowWidget
		from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager

		# Define the set of buttons:
		button_defns = [("Output Folder", lambda _: reveal_in_system_file_manager(curr_active_pipeline.get_output_path())),
				("global pickle", lambda _: reveal_in_system_file_manager(curr_active_pipeline.global_computation_results_pickle_path)),
				("pipeline pickle", lambda _: reveal_in_system_file_manager(curr_active_pipeline.pickle_path)),
				(".h5 export", lambda _: reveal_in_system_file_manager(curr_active_pipeline.h5_export_path)),
				("ViTables .h5 export", lambda _: reveal_in_system_file_manager(curr_active_pipeline.h5_export_path))
			]

		# Create and display the button
		button_executor = JupyterButtonRowWidget(button_defns=button_defns)

	
	"""
	button_list: List
	root_widget: Optional[widgets.HBox]

	def __init__(self, button_defns, defer_display:bool=False):
		self.button_list = []
		self.root_widget = None
		## Build the widget:
		self.build_widget(button_defns)
		# Display if needed
		if not defer_display:
			self.display_buttons()

	def build_widget(self, button_defns):
		## builds the buttons from the definitions:
		self.button_list = []
		for (a_label, a_fn) in button_defns:
			a_btn = widgets.Button(description=a_label)
			a_btn.on_click(a_fn)
			self.button_list.append(a_btn)
		self.root_widget = widgets.HBox(self.button_list)
		

	def display_buttons(self):
		assert self.root_widget is not None
		display(self.root_widget)