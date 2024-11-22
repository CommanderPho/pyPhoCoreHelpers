from typing import Callable, Optional, List, Dict, Union, Any
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display, HTML, Javascript
from pathlib import Path


def render_colors(color_input):
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
    from qtpy.QtGui import QColor, QBrush, QPen

    # color_html = ''.join([f'<div style="width:50px; height:50px; background-color:{color}; margin:5px; display:inline-block;"></div>' for color in color_list])

    # Check if input is a dictionary
    if isinstance(color_input, dict):
        color_items = color_input.items()
    else:
        # Ensure color_input is a list, even if a single color is provided
        if not isinstance(color_input, (list, tuple)):
            color_input = [color_input]
        # Create a list of (label, color) where label is None for lists
        color_items = [(None, color) for color in color_input]

    # Convert colors to hex format if they are QColor objects and prepare HTML
    color_html = ''
    for label, color in color_items:
        if isinstance(color, QColor):
            hex_color = color.name()
        elif isinstance(color, str):
            hex_color = color if color.startswith('#') else f'#{color.lstrip("#")}'
        else:
            raise ValueError("Color must be a QColor, a hex string, or a valid CSS color name.")
        
        if label:
            color_html += f'<div style="display:inline-block; text-align:center; margin:5px;"><div style="width:50px; height:50px; background-color:{hex_color}; margin:5px;"></div><div>{label}</div></div>'
        else:
            color_html += f'<div style="width:50px; height:50px; background-color:{hex_color}; margin:5px; display:inline-block;"></div>'



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
        value=f"<b style='font-size: smaller;'>{a_path}</b>",
        layout=widgets.Layout(width='auto', flex='1 1 auto', margin='2px')
    )


    button_layout = widgets.Layout(flex='0 1 auto', width='auto', min_width='80px', margin='1px') # The button_layout ensures that buttons don't grow and are only as wide as necessary.
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
            # is_file:
            open_button = widgets.Button(description='Open', layout=button_layout, disabled=((not Path(a_path).resolve().exists()) or ((Path(a_path).resolve().is_dir()))), button_style='info', tooltip='Open with default app', icon='external-link-square')
            open_button.on_click(lambda _: open_file_with_system_default(a_path))
            actions_button_list.append(open_button)
        else:
            # is directory
            open_button = widgets.Button(description='Open', layout=button_layout, disabled=((not Path(a_path).resolve().exists()) or ((not Path(a_path).resolve().is_dir()))), button_style='info', tooltip='Open Contents in System Explorer', icon='external-link-square')
            open_button.on_click(lambda _: open_file_with_system_default(a_path))
            actions_button_list.append(open_button)                                            
    

    box_layout_kwargs = (box_layout_kwargs | dict(display='flex', flex_flow='row nowrap',
                                                #    align_items='stretch', width='70%',
                                                    align_items='center', # Vertically align items in the middle
                                                    justify_content='flex-start', # Align items to the start of the container
                                                    width='90%'                                                 
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
    
    # Create the execute in terminal button
    execute_button = widgets.Button(
        description='Execute in Terminal',
        button_style='primary',  # Possible styles: 'success', 'info', 'warning', 'danger' or ''
        tooltip='Execute code in terminal',
        layout={'width': '150px'}  # Adjust the width of the button as needed
    )
    
    # Function to perform the copy to clipboard action
    def on_copy_button_clicked(b):
        payload = f"navigator.clipboard.writeText(`{code_textarea.value}`)"
        js_command = f"eval({payload})"
        display(widgets.HTML(value=f'<img src onerror="{js_command}">'))
    
        # Function to execute code in terminal
    def on_execute_button_clicked(b):
        # Use IPython's system command
        display(Javascript(f'IPython.notebook.kernel.execute("!{code_textarea.value}")'))
        
    # Attach the function to the click event of the button
    copy_button.on_click(on_copy_button_clicked)
    execute_button.on_click(on_execute_button_clicked)
    
    # Use a horizontal box (HBox) to place the button next to the text area
    hbox = widgets.HBox([code_textarea, copy_button, execute_button])
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



def create_tab_widget(display_dict: Dict[str, Any], **tab_kwargs) -> widgets.Tab:
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
    # Inject custom CSS to adjust tab label widths
    display(HTML('''
    <style>
        /* Adjust the tab labels to accommodate longer titles */
        .widget-tab .p-TabBar-tab {
            max-width: none !important;
            min-width: auto !important;
        }
        .widget-tab .p-TabBar-tab .p-TabBar-tabLabel {
            white-space: normal !important;
        }
    </style>
    '''))
        
    # layout =   # Automatically adjust the width to fit titles
    # layout = tab_kwargs.pop('layout', widgets.Layout(width='max-content'))
    layout = tab_kwargs.pop('layout', widgets.Layout(width='auto'))
    tab = widgets.Tab(layout=layout, **tab_kwargs)
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






# ==================================================================================================================== #
# Selection Widgets                                                                                                    #
# ==================================================================================================================== #

# @function_attributes(short_name=None, tags=['widget', 'jupyter', 'ipywidgets'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-22 09:14', related_items=[])
def multi_checkbox_widget(options_dict):
    """ Widget with a search field and lots of checkboxes
    
    import ipywidgets as widgets
    from pyphocorehelpers.gui.Jupyter.simple_widgets import multi_checkbox_widget

    def f(**kwargs):
        print(kwargs)

    options_dict = {
        x: widgets.Checkbox(
            description=x, 
            value=False,
            style={"description_width":"0px"}
        ) for x in ['hello','world']
    }
    ui = multi_checkbox_widget(options_dict)
    out = widgets.interactive_output(f, options_dict)
    display(widgets.HBox([ui, out]))

    """
    search_widget = widgets.Text()
    output_widget = widgets.Output()
    options = [x for x in options_dict.values()]
    options_layout = widgets.Layout(
        overflow='auto',
        border='1px solid black',
        width='300px',
        height='300px',
        flex_flow='column',
        display='flex'
    )
    
    #selected_widget = wid.Box(children=[options[0]])
    options_widget = widgets.VBox(options, layout=options_layout)
    #left_widget = wid.VBox(search_widget, selected_widget)
    multi_select = widgets.VBox([search_widget, options_widget])

    @output_widget.capture()
    def on_checkbox_change(change):
        
        selected_recipe = change["owner"].description
        #print(options_widget.children)
        #selected_item = wid.Button(description = change["new"])
        #selected_widget.children = [] #selected_widget.children + [selected_item]
        options_widget.children = sorted([x for x in options_widget.children], key = lambda x: x.value, reverse = True)
        
    for checkbox in options:
        checkbox.observe(on_checkbox_change, names="value")

    # Wire the search field to the checkboxes
    @output_widget.capture()
    def on_text_change(change):
        search_input = change['new']
        if search_input == '':
            # Reset search field
            new_options = sorted(options, key = lambda x: x.value, reverse = True)
        else:
            # Filter by search field using difflib.
            #close_matches = difflib.get_close_matches(search_input, list(options_dict.keys()), cutoff=0.0)
            close_matches = [x for x in list(options_dict.keys()) if str.lower(search_input.strip('')) in str.lower(x)]
            new_options = sorted(
                [x for x in options if x.description in close_matches], 
                key = lambda x: x.value, reverse = True
            ) #[options_dict[x] for x in close_matches]
        options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')
    display(output_widget)
    return multi_select
