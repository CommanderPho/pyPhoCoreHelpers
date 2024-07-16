from typing import Callable, Optional, List, Dict, Union, Any
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display, HTML, Javascript
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
    # right_label = widgets.Label(a_path, layout=widgets.Layout(width='auto', flex='1 1 auto', margin='2px'))
    # Change the style of 'right_label' to make it bolder and larger using HTML tags in the value
    right_label = widgets.HTML(
        value=f"<b style='font-size: larger;'>{a_path}</b>",
        layout=widgets.Layout(width='auto', flex='1 1 auto', margin='2px')
    )


    button_layout = widgets.Layout(flex='0 1 auto', width='auto', margin='1px') # The button_layout ensures that buttons don't grow and are only as wide as necessary.
    # right_label_layout = widgets.Layout(flex='1 1 auto', min_width='0px', width='auto') # We use the flex property in the right_label_layout to let the label grow and fill the space, but it can also shrink if needed (flex='1 1 auto'). We set a min_width so it doesn't get too small and width='auto' to let it size based on content
    
    actions_button_list = []
    copy_to_clipboard_button = widgets.Button(description='Copy', layout=button_layout, disabled=(not Path(a_path).resolve().exists()), button_style='info', tooltip='Copy to Clipboard', icon='clipboard') # , icon='folder-tree'
    copy_to_clipboard_button.on_click(lambda _: copy_to_clipboard(str(a_path)))
    actions_button_list.append(copy_to_clipboard_button)

    reveal_button = widgets.Button(description='Reveal', layout=button_layout, disabled=(not Path(a_path).resolve().exists()), button_style='info', tooltip='Reveal in System Explorer', icon='folder-open-o')
    reveal_button.on_click(lambda _: reveal_in_system_file_manager(a_path))
    actions_button_list.append(reveal_button)

    if has_valid_file:
        is_dir = resolved_path.is_dir()
        if not is_dir:
            open_button = widgets.Button(description='Open', layout=button_layout, disabled=((not Path(a_path).resolve().exists()) or ((Path(a_path).resolve().is_dir()))), button_style='info', tooltip='Open with default app', icon='external-link-square')
            open_button.on_click(lambda _: open_file_with_system_default(a_path))
            actions_button_list.append(open_button)

    box_layout_kwargs = (box_layout_kwargs | dict(display='flex', flex_flow='row nowrap',
                                                #    align_items='stretch', width='70%',
                                                    align_items='center', # Vertically align items in the middle
                                                    justify_content='flex-start', # Align items to the start of the container
                                                    width='70%'                                                 
                                                   ))
    box_layout = widgets.Layout(**box_layout_kwargs)
    hbox = widgets.Box(children=[left_label, right_label, *actions_button_list], layout=box_layout)
    return hbox




def simple_path_display_widget(a_path: Union[Path, str]):
    """ Returns a simple clickable Path that works on Windows and for paths containing spaces.
    
    Call like:
        from pyphocorehelpers.gui.Jupyter.simple_widgets import simple_path_display_widget, _build_file_link_from_path
        simple_path_display_widget(r"C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2024-01-17/kdiba/gor01/one/2006-6-08_14-26-15/plot_all_epoch_bins_marginal_predictions_Laps all_epoch_binned Marginals.png")
        
    NOTE: could use from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path
    
    """
    def _subfn_build_file_link_from_path(a_path: Union[Path, str]) -> str:
        # if not isinstance(a_path, str):
        #     a_path = str(a_path)
        if not isinstance(a_path, Path):
            a_path = Path(a_path).resolve() # we need a Path
        a_path_url_str: str = a_path.as_uri() # returns a string like "file:///C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2024-01-17/kdiba/gor01/one/2006-6-08_14-26-15/plot_all_epoch_bins_marginal_predictions_Laps%20all_epoch_binned%20Marginals.png"
        return f'<a href="{a_path_url_str}" target="_blank">{a_path_url_str}</a>'
        # return f'<a href="file://{a_path_url_str}" target="_blank">{a_path_url_str}</a>'

    # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
    file_link: str = _subfn_build_file_link_from_path(a_path)
    return HTML(file_link)



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




def build_dropdown_selection_widget(all_items: List[Path], on_user_update_item_selection: Callable):
    """ Builds a dropdown list that allows the user to select from a list of items, such as contexts.
        from pyphocorehelpers.gui.Jupyter.simple_widgets import build_dropdown_selection_widget

        active_slected_context = included_session_contexts[0]

        def selected_context_changed_callback(context_str, context_data):
            global active_slected_context
            print(f"You selected {context_str}:{context_data}")
            active_slected_context = context_data # update the selected context

        context_selection_dropdown = build_dropdown_selection_widget(included_session_contexts, on_user_update_item_selection=selected_context_changed_callback)
        context_selection_dropdown
    
    """
    item_list = all_items.copy()
    assert len(item_list) > 0, f"item_list is empty!"
    # global_data_selected_item = item_list[0]        

    def selected_context_changed_callback(change):
        # global global_data_selected_item
        selected_item = change['new']
        print(f"You selected {selected_item}")
        # Run your function or cell code here
        on_user_update_item_selection(str(selected_item), selected_item)


    # item_list = ['Item 1', 'Item 2', 'Item 3']
    # item_list = included_session_contexts.copy()
    # item_list = all_items.copy()
    context_selection_dropdown = widgets.Dropdown(options=item_list, description='Select Context:', layout=widgets.Layout(width='auto'))
    context_selection_dropdown.style = {'description_width': 'initial'} # Increase dropdown width with CSS
    context_selection_dropdown.observe(selected_context_changed_callback, 'value')

    ## Call the user function with the first extant path:
    on_user_update_item_selection(str(item_list[0]), item_list[0])
    
    return context_selection_dropdown


def code_block_widget(contents: str, label: str="Code:"):
    """
    Create a code block widget with a copy-to-clipboard button.
    
    Parameters:
    contents (str): The initial text/content to display in the code block.
    label (str): The label for the code block textarea.

    Usage:

        from pyphocorehelpers.gui.Jupyter.simple_widgets import code_block_widget

        # Create and display the code block widget
        slurm_code_block = code_block_widget(initial_code, label="Python Code:")
    
    """
    # Create the code block text area widget
    code_textarea = widgets.Textarea(
        value=contents,
        placeholder='Type code here',
        description=label,
        disabled=False,
        layout=widgets.Layout(width='100%', height='200px')  # Adjust the size as needed
    )
    
    # Create the copy-to-clipboard button
    copy_button = widgets.Button(
        description='Copy to Clipboard',
        button_style='success',  # Possible styles: 'success', 'info', 'warning', 'danger' or ''
        tooltip='Copy code to clipboard',
        layout={'width': '150px'}  # Adjust the width of the button as needed
    )
    
    # Function to perform the copy to clipboard action
    def on_copy_button_clicked(b):
        payload = f"navigator.clipboard.writeText(`{code_textarea.value}`)"
        js_command = f"eval({payload})"
        display(widgets.HTML(value=f'<img src onerror="{js_command}">'))
    
    # Attach the function to the click event of the button
    copy_button.on_click(on_copy_button_clicked)
    
    # Use a horizontal box (HBox) to place the button next to the text area
    hbox = widgets.HBox([code_textarea, copy_button])
    display(hbox)
    return hbox



def filesystem_path_folder_contents_widget(a_path: Union[Path, str], on_file_open=None):
    """ Returns a simple clickable Path that works on Windows and for paths containing spaces.
    
    Call like:
        from pyphocorehelpers.gui.Jupyter.simple_widgets import filesystem_path_folder_contents_widget

        curr_collected_outputs_folder = Path(output_path_dicts['neuron_replay_stats_table']['.csv']).resolve().parent        
        filesystem_path_folder_contents_widget(curr_collected_outputs_folder)
    
    """
    import solara # `pip install "solara[assets]`

    if not isinstance(a_path, Path):
        a_path = Path(a_path).resolve() # we need a Path
    assert a_path.exists(), f'a_path: "{a_path} does not exist!"'


    if on_file_open is None:
        on_file_open = print

    return widgets.VBox(
            children=[
                solara.FileBrowser.widget(directory=a_path, on_file_open=on_file_open)
            ]
        )



def create_tab_widget(display_dict: Dict[str, Any]) -> widgets.Tab:
    """
    Creates an ipywidgets Tab widget that allows tabbing between multiple display items.

    Args:
        display_dict (Dict[str, Any]): A dictionary where keys are titles (str) and values are display items (Any).

    Returns:
        widgets.Tab: An ipywidgets Tab widget.
        
    Usage:
        from pyphocorehelpers.gui.Jupyter.simple_widgets import create_tab_widget
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})

        tab_widget = create_tab_widget({"DataFrame 1": df1, "DataFrame 2": df2})
        display(tab_widget)

    """
    tab = widgets.Tab()
    children = []
    titles = list(display_dict.keys())
    items = list(display_dict.values())
    
    for item in items:
        children.append(widgets.Output())
    
    tab.children = children
    
    for i, title in enumerate(titles):
        tab.set_title(i, title)
    
    for i, item in enumerate(items):
        with children[i]:
            display(item)
    
    return tab






