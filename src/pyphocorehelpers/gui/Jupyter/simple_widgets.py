from typing import Callable, Optional, List, Dict
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display, HTML
from pathlib import Path


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



# ==================================================================================================================== #
# Filesystem Paths                                                                                                     #
# ==================================================================================================================== #

def fullwidth_path_widget(a_path, file_name_label: str="session path:", box_layout_kwargs=None):
    """ displays a simple file path and a reveal button that shows it. 
     
     from pyphocorehelpers.gui.Jupyter.simple_widgets import fullwidth_path_widget
     
    """
    from pyphocorehelpers.Filesystem.path_helpers import open_file_with_system_default
    from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager
    from pyphocorehelpers.programming_helpers import copy_to_clipboard

    box_layout_kwargs = box_layout_kwargs or {} # empty
    
    has_valid_file = False
    resolved_path: Optional[Path] = None

    if a_path is None:
        a_path = "<None>"
    else:
        if not isinstance(a_path, str):
            a_path = str(a_path)       
        resolved_path = Path(a_path).resolve()             
        has_valid_file = resolved_path.exists()
        is_dir = resolved_path.is_dir()
    
    
    left_label = widgets.Label(file_name_label, layout=widgets.Layout(width='auto'))
    right_label = widgets.Label(a_path, layout=widgets.Layout(width='auto', flex='1 1 auto', margin='2px'))

    actions_button_list = []
    copy_to_clipboard_button = widgets.Button(description='Copy', layout=widgets.Layout(flex='0 1 auto', width='auto', margin='1px'), disabled=(not Path(a_path).resolve().exists()), button_style='info', tooltip='Copy to Clipboard') # , icon='folder-tree'
    copy_to_clipboard_button.on_click(lambda _: copy_to_clipboard(str(a_path)))
    actions_button_list.append(copy_to_clipboard_button)

    reveal_button = widgets.Button(description='Reveal', layout=widgets.Layout(flex='0 1 auto', width='auto', margin='1px'), disabled=(not Path(a_path).resolve().exists()), button_style='info', tooltip='Reveal in System Explorer', icon='folder-tree')
    reveal_button.on_click(lambda _: reveal_in_system_file_manager(a_path))
    actions_button_list.append(reveal_button)

    if has_valid_file:
        is_dir = resolved_path.is_dir()
        if not is_dir:
            open_button = widgets.Button(description='Open', layout=widgets.Layout(flex='0 1 auto', width='auto', margin='1px'), disabled=((not Path(a_path).resolve().exists()) or ((Path(a_path).resolve().is_dir()))), button_style='info', tooltip='Open with default app', icon='folder-tree')
            open_button.on_click(lambda _: open_file_with_system_default(a_path))
            actions_button_list.append(open_button)

    box_layout_kwargs = (box_layout_kwargs | dict(display='flex', flex_flow='row', align_items='stretch', width='70%'))
    box_layout = widgets.Layout(**box_layout_kwargs)
    hbox = widgets.Box(children=[left_label, right_label, *actions_button_list], layout=box_layout)
    return hbox





def build_global_data_root_parent_path_selection_widget(all_paths: List[Path], on_user_update_path_selection: Callable):
    """ 
    from pyphocorehelpers.gui.Jupyter.simple_widgets import build_global_data_root_parent_path_selection_widget
    
    all_paths = [Path(r'/home/halechr/turbo/Data'), Path(r'W:\Data'), Path(r'/home/halechr/FastData'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data')]
    global_data_root_parent_path = extant_paths[0]
    def on_user_update_path_selection(new_path: Path):
        global global_data_root_parent_path
        new_global_data_root_parent_path = new_path.resolve()
        global_data_root_parent_path = new_global_data_root_parent_path
        print(f'global_data_root_parent_path changed to {global_data_root_parent_path}')
        assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
                
    global_data_root_parent_path_widget = build_global_data_root_parent_path_selection_widget(all_paths, on_user_update_path_selection)
    global_data_root_parent_path_widget
    
    """
    extant_paths = [a_path for a_path in all_paths if a_path.exists()]
    assert len(extant_paths) > 0, f"NO EXTANT PATHS FOUND AT ALL!"
    global_data_root_parent_path = extant_paths[0]        


    # widgets.ToggleButtons
    global_data_root_parent_path_widget = widgets.ToggleButtons(
                                            options=extant_paths,
                                            description='Data Root:',
                                            disabled=False,
                                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                            tooltip='global_data_root_parent_path',
                                            layout=widgets.Layout(width='auto'),
                                            style={"button_width": 'max-content'}, # "190px"
                                        #     icon='check'
                                        )

    def on_global_data_root_parent_path_selection_change(change):
        global global_data_root_parent_path
        new_global_data_root_parent_path = Path(str(change['new'])).resolve()
        global_data_root_parent_path = new_global_data_root_parent_path
        print(f'global_data_root_parent_path changed to {global_data_root_parent_path}')
        assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
        on_user_update_path_selection(new_global_data_root_parent_path)
        

    global_data_root_parent_path_widget.observe(on_global_data_root_parent_path_selection_change, names='value')
    ## Call the user function with the first extant path:
    on_user_update_path_selection(extant_paths[0])

    return global_data_root_parent_path_widget


