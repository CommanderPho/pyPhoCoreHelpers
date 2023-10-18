import ipywidgets as widgets
from IPython.core.display import display, HTML

def render_colors(color_list):
    """ Renders a simple list of colors for visual previewing
    Usage:
    
		from pyphocorehelpers.gui.Jupyter.simple_widgets import render_colors

		render_colors(color_list)

    Advanced Usage:
    
		# Define the list of colors you want to display
		# color_list = ['red', 'blue', 'green', '#FFA500', '#800080']
		color_list = _plot_backup_colors.neuron_colors_hex

		# Create a button to trigger the color rendering
		button = widgets.Button(description="Show Colors")

		# Define what happens when the button is clicked
		def on_button_click(b):
			render_colors(color_list)

		button.on_click(on_button_click)

		# Display the button
		button
        
    """
    color_html = ''.join([f'<div style="width:50px; height:50px; background-color:{color}; margin:5px; display:inline-block;"></div>' for color in color_list])
    display(HTML(color_html))


