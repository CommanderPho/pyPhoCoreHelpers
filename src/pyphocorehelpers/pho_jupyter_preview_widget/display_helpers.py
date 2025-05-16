from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, NewType, TypeVar
import IPython
import IPython.display
from nptyping import NDArray

import numpy as np
import pandas as pd
# import dask.array as da
# from pho_jupyter_preview_widget.array_shape_display import array_repr_html
# from pho_jupyter_preview_widget.array_shape_display.array_shape_display import array_repr_html
from pyphocorehelpers.pho_jupyter_preview_widget.array_shape_display.array_shape_display import array_repr_html

import ipykernel # ip: "ipykernel.zmqshell.ZMQInteractiveShell)" = IPython.get_ipython()
from IPython.display import display, HTML, Javascript
from ipywidgets import widgets, VBox, HBox

from io import BytesIO
import base64


import matplotlib.pyplot as plt

from pyphocorehelpers.print_helpers import render_scrollable_colored_table_from_dataframe, render_scrollable_colored_table


# ==================================================================================================================== #
# 2024-05-30 - Custom Formatters                                                                                       #
# ==================================================================================================================== #

    
def array_preview_with_shape(arr):
    """ Text-only Represntation that prints np.shape(arr) 
    
        from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import array_preview_with_shape

        # Register the custom display function for numpy arrays
        import IPython
        ip = IPython.get_ipython()
        ip.display_formatter.formatters['text/html'].for_type(np.ndarray, array_preview_with_shape) # only registers for NDArray

        # Example usage
        arr = np.random.rand(3, 4)
        display(arr)

    """
    if isinstance(arr, np.ndarray):
        display(HTML(f"<pre>array{arr.shape} of dtype {arr.dtype}</pre>"))
    elif isinstance(arr, (list, tuple)):
        display(HTML(f"<pre>native-python list {len(arr)}</pre>"))
    elif isinstance(arr, pd.DataFrame):
        display(HTML(f"<pre>DataFrame with {len(arr)} rows and {len(arr.columns)} columns</pre>"))
    else:
        raise ValueError("The input is not a NumPy array.")


def array_preview_with_graphical_shape_repr_html(arr):
    """Generate an HTML representation for a NumPy array, similar to Dask.
        
    from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import array_preview_with_graphical_shape_repr_html
    
    # Register the custom display function for NumPy arrays
    import IPython
    ip = IPython.get_ipython()
    ip.display_formatter.formatters['text/html'].for_type(np.ndarray, lambda arr: array_preview_with_graphical_shape_repr_html(arr))

    # Example usage
    arr = np.random.rand(3, 4)
    display(arr)


    arr = np.random.rand(9, 64)
    display(arr)

    arr = np.random.rand(9, 64, 4)
    display(arr)

    """
    if isinstance(arr, np.ndarray):
        # arr = da.array(arr)
        arr = array_repr_html(arr)
        return display(arr)
        # shape_str = ' &times; '.join(map(str, arr.shape))
        # dtype_str = arr.dtype
        # return f"<pre>array[{shape_str}] dtype={dtype_str}</pre>"
    else:
        raise ValueError("The input is not a NumPy array.")



# Generate heatmap
class MatplotlibToIPythonWidget:
    """ 
    
    from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import MatplotlibToIPythonWidget
    
    MatplotlibToIPythonWidget.matplotlib_fig_to_ipython_HTML(fig=fig)
    
    """
    @classmethod
    def _matplotlib_fig_to_bytes(cls, fig) -> Optional[BytesIO]: # , omission_indices: list = None
        """ 
        
        #TODO 2024-08-16 04:05: - [ ] Make non-interactive and open in the background

        from neuropy.utils.matplotlib_helpers import matplotlib_configuration
        with matplotlib_configuration(is_interactive=False, backend='AGG'):
            # Perform non-interactive Matplotlib operations with 'AGG' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode with AGG Backend')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

                
        import matplotlib
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        _bak_rcParams = mpl.rcParams.copy()

        matplotlib.use('Qt5Agg')
        # %matplotlib inline
        # %matplotlib auto


        # _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
        _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

            
        """
        try:                
            buf = BytesIO()            
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
        except BaseException as err:
            # SystemError: tile cannot extend outside image
            print(f'ERROR: Encountered error while convert matplotlib fig: {fig} to bytes:\n\terr: {err}')
            buf = None
        # finally:
        #     plt.close()        
        
        return buf


    @classmethod
    def matplotlib_fig_to_ipython_img(cls, fig, **img_kwargs) -> Optional[widgets.Image]:
        img_kwargs = dict(width=None, height=img_kwargs.get('height', 100), format='png') | img_kwargs
        buf = cls._matplotlib_fig_to_bytes(fig)
        if buf is not None:
            # Create an IPython Image object
            img = widgets.Image(data=buf.getvalue(), **img_kwargs)
            return img
        else:
            return None
    

    # Convert to ipywidgets Image
    @classmethod
    def matplotlib_fig_to_ipython_HTML(cls, fig, horizontal_layout=True, **kwargs) -> str:
        """ Generate an HTML representation for a NumPy array with a Dask shape preview and a thumbnail heatmap
        
            from pyphocorehelpers.pho_jupyter_preview_widget.pho_jupyter_preview_widget.display_helpers import array_preview_with_heatmap_repr_html

            # Register the custom display function for numpy arrays
            import IPython
            ip = IPython.get_ipython()
            ip.display_formatter.formatters['text/html'].for_type(np.ndarray, array_preview_with_heatmap) # only registers for NDArray

            # Example usage
            arr = np.random.rand(3, 4)
            display(arr)

        """

        # print(f'WARN: n_dim: {n_dim} greater than 2 is unsupported!')
        # # from pyphocorehelpers.plotting.media_output_helpers import get_array_as_image_stack
        # # #TODO 2024-08-13 05:05: - [ ] use get_array_as_image_stack to render the 3D array
        # message = f"Heatmap Err: n_dim: {n_dim} greater than 2 is unsupported!"
        # heatmap_html = f"""
        # <div style="text-align: center; padding: 20px; border: 1px solid #ccc;">
        #     <p style="font-size: 16px; color: red;">{message}</p>
        # </div>
        # """

        out_image = cls.matplotlib_fig_to_ipython_img(fig, **kwargs)
        if (out_image is not None):
            orientation = "row" if horizontal_layout else "column"
            ## Lays out side-by-side:
            # Convert the IPython Image object to a base64-encoded string
            out_image_data = out_image.data
            b64_image = base64.b64encode(out_image_data).decode('utf-8')
            # Create an HTML widget for the heatmap
            fig_size_format_str: str = ''
            width = kwargs.get('width', None)
            if (width is not None) and (width > 0):
                fig_size_format_str = fig_size_format_str + f'width="{width}" '
            height = kwargs.get('height', None)
            if (height is not None) and (height > 0):
                fig_size_format_str = fig_size_format_str + f'height="{height}" '
            
            fig_image_html = f'<img src="data:image/png;base64,{b64_image}" {fig_size_format_str}style="background:transparent;"/>' #  width="{ndarray_preview_config.heatmap_thumbnail_width}"

        else:
            # getting image failed:
            # Create an HTML widget for the heatmap
            message = "Heatmap Err"
            fig_image_html = f"""
            <div style="text-align: center; padding: 20px; border: 1px solid #ccc;">
                <p style="font-size: 16px; color: red;">{message}</p>
            </div>
            """

            
        # Combine both HTML representations
        if horizontal_layout:
            combined_html = f"""
            <div style="display: flex; flex-direction: row; align-items: flex-start;">
                <div>{fig_image_html}</div>
            </div>
            """
        else:
            combined_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div>{fig_image_html}</div>
                <div style="margin-top: 10px;">
                </div>
            </div>
            """
        return combined_html


    
def _subfn_create_heatmap(data: NDArray, brokenaxes_kwargs=None) -> Optional[BytesIO]: # , omission_indices: list = None
    """ 
    
    #TODO 2024-08-16 04:05: - [ ] Make non-interactive and open in the background

    from neuropy.utils.matplotlib_helpers import matplotlib_configuration
    with matplotlib_configuration(is_interactive=False, backend='AGG'):
        # Perform non-interactive Matplotlib operations with 'AGG' backend
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Non-interactive Mode with AGG Backend')
        plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

            
    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    _bak_rcParams = mpl.rcParams.copy()

    matplotlib.use('Qt5Agg')
    # %matplotlib inline
    # %matplotlib auto


    # _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
    _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

        
    """
    if (data.ndim < 2):
        data = np.atleast_2d(data)
        # fix issues with 1D data like `TypeError: Invalid shape (58,) for image data`
    
    import matplotlib.pyplot as plt
    
    try:
        imshow_shared_kwargs = {
            'origin': 'lower',
        }

        active_cmap = 'viridis'
        fig = plt.figure(figsize=(3, 3), num='_jup_backend')
        ax = fig.add_subplot(111)
        ax.imshow(data, cmap=active_cmap, **imshow_shared_kwargs)
        ax.axis('off')
            
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
    except SystemError as err:
        # SystemError: tile cannot extend outside image
        print(f'ERROR: Encountered error while plotting heatmap:\n\terr: {err}')
        print(f'\tnp.shape(data): {np.shape(data)}\n\tdata: {data}')
        buf = None

    finally:
        plt.close()        
    
    return buf

# Convert to ipywidgets Image
def _subfn_display_heatmap(data: NDArray, brokenaxes_kwargs=None, **img_kwargs) -> Optional[IPython.core.display.Image]:
    """ Renders a small thumbnail Image of a heatmap array
    
    """
    img_kwargs = dict(width=None, height=img_kwargs.get('height', 100), format='png') | img_kwargs
    buf = _subfn_create_heatmap(data, brokenaxes_kwargs=brokenaxes_kwargs)
    if buf is not None:
        # Create an IPython Image object
        img = IPython.core.display.Image(data=buf.getvalue(), **img_kwargs) # IPython.core.display.Image
        return img
    else:
        return None


# @function_attributes(short_name='array2str', tags=['array', 'formatting', 'fix'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-27 07:45')
def smart_array2string(arr: NDArray, disable_readible_format: bool=False, separator=',', **kwargs) -> str:
    """ Drop-in replacement for `np.array2string` which consistently handles spaces. np.array2string automatically tries to format arrays in a way that is readable for matricies of data, but it uses the same formatting rules for 1D arrays, resulting in inconsistent numbers of spaces between elements. This function fixes that.
    if disable_readible_format is False, nothing special is done.
    """
    if disable_readible_format and (np.ndim(arr) == 1):
        ## np.array2string automatically tries to format arrays in a way that is readable for matricies of data, but it uses the same formatting rules for 1D arrays, resulting in inconsistent numbers of spaces between elements. This function fixes that.
        return f'{separator} '.join([v.strip(' ') for v in np.array2string(arr, separator=separator, **kwargs).split(separator)]).replace('[ ', '[').replace(' ]', ']')
    else:
        return np.array2string(arr, separator=separator, **kwargs)
    

# ==================================================================================================================== #
# Main formatting function                                                                                             #
# ==================================================================================================================== #
def single_NDArray_array_preview_with_heatmap_repr_html(arr, include_shape: bool=True, horizontal_layout=True, include_plaintext_repr:bool=False, **kwargs):
    """ Generate an HTML representation for a NumPy array with a Dask shape preview and a thumbnail heatmap
    
        from pho_jupyter_preview_widget.pho_jupyter_preview_widget.display_helpers import array_preview_with_heatmap_repr_html

        # Register the custom display function for numpy arrays
        import IPython
        ip = IPython.get_ipython()
        ip.display_formatter.formatters['text/html'].for_type(np.ndarray, array_preview_with_heatmap) # only registers for NDArray

        # Example usage
        arr = np.random.rand(3, 4)
        display(arr)

    """
    max_allowed_arr_elements: int = 10000

    if isinstance(arr, np.ndarray):
        
        n_dim: int = np.ndim(arr)
        if n_dim > 2:
            print(f'WARN: n_dim: {n_dim} greater than 2 is unsupported!')
            # from pyphocorehelpers.plotting.media_output_helpers import get_array_as_image_stack
            # #TODO 2024-08-13 05:05: - [ ] use get_array_as_image_stack to render the 3D array
            message = f"Heatmap Err: n_dim: {n_dim} greater than 2 is unsupported!"
            heatmap_html = f"""
            <div style="text-align: center; padding: 20px; border: 1px solid #ccc;">
                <p style="font-size: 16px; color: red;">{message}</p>
            </div>
            """

        else:
            ## n_dim == 2
            if np.shape(arr)[0] > max_allowed_arr_elements: 
                # truncate 
                arr = arr[max_allowed_arr_elements:]
            
            heatmap_image = _subfn_display_heatmap(arr, **kwargs)
            if (heatmap_image is not None):
                orientation = "row" if horizontal_layout else "column"
                ## Lays out side-by-side:
                # Convert the IPython Image object to a base64-encoded string
                heatmap_image_data = heatmap_image.data
                b64_image = base64.b64encode(heatmap_image_data).decode('utf-8')
                # Create an HTML widget for the heatmap
                heatmap_size_format_str: str = ''
                width = kwargs.get('width', None)
                if (width is not None) and (width > 0):
                    heatmap_size_format_str = heatmap_size_format_str + f'width="{width}" '
                height = kwargs.get('height', None)
                if (height is not None) and (height > 0):
                    heatmap_size_format_str = heatmap_size_format_str + f'height="{height}" '
                
                heatmap_html = f'<img src="data:image/png;base64,{b64_image}" {heatmap_size_format_str}style="background:transparent;"/>' #  width="{ndarray_preview_config.heatmap_thumbnail_width}"

            else:
                # getting image failed:
                # Create an HTML widget for the heatmap
                message = "Heatmap Err"
                heatmap_html = f"""
                <div style="text-align: center; padding: 20px; border: 1px solid #ccc;">
                    <p style="font-size: 16px; color: red;">{message}</p>
                </div>
                """

        # height="{height}"
        dask_array_widget_html = ""
        plaintext_html = ""
        
        if include_shape:
            # dask_array_widget: widgets.HTML = widgets.HTML(value=da.array(arr)._repr_html_())
            # dask_array_widget: widgets.HTML = widgets.HTML(value=array_repr_html(arr)) ## use new custom `array_repr_html` function
            dask_array_widget: widgets.HTML = widgets.HTML(value=array_repr_html(arr.shape, None, arr.dtype)) ## use new custom `array_repr_html` function

            dask_array_widget_html: str = dask_array_widget.value
            dask_array_widget_html = f"""
                <div style="margin-left: 10px;">
                    {dask_array_widget_html}
                </div>
            """

        if include_plaintext_repr:                
            # plaintext_repr = smart_array2string(arr, edgeitems=3, threshold=5)  # Adjust these parameters as needed
            plaintext_repr = smart_array2string(arr)
            plaintext_html = f"<pre>{plaintext_repr}</pre>"
            plaintext_html = f"""
                <div style="margin-left: 10px;">
                    {plaintext_html}
                </div>
            """
            
        # Combine both HTML representations
        if horizontal_layout:
            ## vertical layout:
            combined_html = f"""
            <div style="display: flex; flex-direction: row; align-items: flex-start;">
                <div>{heatmap_html}</div>
                {dask_array_widget_html}
                {plaintext_html}
            </div>
            """
        else:
            ## vertical layout:
            combined_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div>{heatmap_html}</div>
                <div style="margin-top: 10px;">
                    {dask_array_widget_html}
                    {plaintext_html}
                </div>
            </div>
            """
        return combined_html

    else:
        raise ValueError("The input is not a NumPy array.")


def array_preview_with_heatmap_repr_html(arr_or_list, include_shape: bool=True, horizontal_layout=True, include_plaintext_repr:bool=False, **kwargs):
    """
    Generates an HTML representation for a single numpy array or a list of numpy arrays.
    """
    # output_fn = HTML
    output_fn = str
    
    def format_single_array(arr):
        """ captures: include_shape, horizontal_layout, include_plaintext_repr, **kwargs """
        # Use your existing logic for single numpy array heatmap representation
        # Assuming this logic generates an HTML string for an array heatmap
        return single_NDArray_array_preview_with_heatmap_repr_html(arr, include_shape=include_shape, horizontal_layout=horizontal_layout, include_plaintext_repr=include_plaintext_repr, **kwargs)
    

    if isinstance(arr_or_list, list):
        if all(isinstance(v, np.ndarray) for v in arr_or_list):
            # Handle list of numpy arrays
            # formatted_arrays = [format_single_array(arr) for arr in arr_or_list] # not sure if we want to show the heatmap for each array, probably not.
            # return output_fn("<ul>" + "".join(f"<li>{fa}</li>" for fa in formatted_arrays) + "</ul>") ## do I want to return the HTML or just the raw string?
            formatted_arrays = [smart_array2string(arr) for arr in arr_or_list]
            plaintext_repr: str = ', '.join(formatted_arrays)
            plaintext_html = f"<pre>{plaintext_repr}</pre>"
            plaintext_html = f"""
                <div style="margin-left: 10px;">
                    {plaintext_html}
                </div>
            """
            return output_fn(plaintext_html)

        else:
            # If the list contains non-ndarray types, fallback to repr                
            return output_fn(f"{repr(arr_or_list)}") # return default repr
        
            # return output_fn(f"<div>Unsupported list elements: {repr(arr_or_list)}</div>")
    elif isinstance(arr_or_list, np.ndarray):
        # Handle single numpy array
        return output_fn(format_single_array(arr_or_list))
    else:
        # Fallback for unsupported types
        return output_fn(f"<div>Unsupported type: {type(arr_or_list)}</div>")


# ---------------------------------------------------------------------------- #
#                       Jupyter Datatype Printing Helpers                      #
# ---------------------------------------------------------------------------- #

def array_repr_with_graphical_shape(ip: "ipykernel.zmqshell.ZMQInteractiveShell") -> "ipykernel.zmqshell.ZMQInteractiveShell":
    """Generate an HTML representation for a NumPy array, similar to Dask.
        
    from preferences_helpers import array_graphical_shape
    from pho_jupyter_preview_widget.display_helpers import array_preview_with_graphical_shape_repr_html
    
    # Register the custom display function for NumPy arrays
    import IPython
    
    ip: "ipykernel.zmqshell.ZMQInteractiveShell)" = IPython.get_ipython()


    """
    from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import array_preview_with_graphical_shape_repr_html
    # Register the custom display function for NumPy arrays
    ip.display_formatter.formatters['text/html'].for_type(np.ndarray, lambda arr: array_preview_with_graphical_shape_repr_html(arr))
    return ip


def array_repr_with_graphical_preview(ip: "ipykernel.zmqshell.ZMQInteractiveShell", include_shape: bool=True, horizontal_layout:bool=True, include_plaintext_repr:bool=True, height:int=50, width:Optional[int]=None) -> "ipykernel.zmqshell.ZMQInteractiveShell":
    """Generate an HTML representation for a NumPy array with a Dask shape preview and a thumbnail heatmap
    
    """
    from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import array_preview_with_heatmap_repr_html

    # def format_single_array(arr):
    #     # Your existing logic to render a single np.ndarray
    #     return f"<div>{smart_array2string(arr, precision=3, separator=', ', suppress_small=True)}</div>"

    # if isinstance(arr_or_list, list) and all(isinstance(v, np.ndarray) for v in arr_or_list):
    #     # Handle list of np.ndarray
    #     formatted_arrays = [format_single_array(arr) for arr in arr_or_list]
    #     return HTML("<ul>" + "".join(f"<li>{fa}</li>" for fa in formatted_arrays) + "</ul>")
    # elif isinstance(arr_or_list, np.ndarray):
    #     # Handle single np.ndarray
    #     return HTML(format_single_array(arr_or_list))
    # else:
    #     # Fallback for other types
    #     return HTML(f"<div>Unsupported type: {type(arr_or_list)}</div>")


    # Register the custom display function for NumPy arrays
    ip.display_formatter.formatters['text/html'].for_type(np.ndarray, lambda arr: array_preview_with_heatmap_repr_html(arr, include_shape=include_shape, horizontal_layout=horizontal_layout, include_plaintext_repr=include_plaintext_repr, height=height, width=width))
    
    ip.display_formatter.formatters['text/html'].for_type(list, lambda lst: array_preview_with_heatmap_repr_html(lst))


    # ## Plain-text type representation can be suppressed like:
    if include_plaintext_repr:
        # Override text formatter to prevent plaintext representation
        ## SIDE-EFFECT: messes up printing NDARRAYs embedded in lists, dicts, other objects, etc. It seems that when rendering these they use the 'text/plain' representations
        ip.display_formatter.formatters['text/plain'].for_type(
            np.ndarray, 
            lambda arr, p, cycle: None
        )
    
    return ip


#TODO 2024-08-07 13:02: - [ ] Finish custom plaintext formatting
# # Register the custom display function for NumPy arrays
# import IPython
# ip = IPython.get_ipython()

# def format_list_of_ndarrays(obj, p, cycle):
#     if all(isinstance(x, np.ndarray) for x in obj):
#         return "[" + ", ".join(smart_array2string(a) for a in obj) + "]"
#     else:
#         # Fallback to the original formatter
#         return p.text(repr(obj))
#         # # Use the existing formatter if present, or default to repr
#         # existing_formatter = ip.display_formatter.formatters['text/plain'].lookup_by_type(list)
#         # return existing_formatter(obj, p, cycle) if existing_formatter else repr(obj)
    

# # ip.display_formatter.formatters['text/plain'].for_type(
# #     List[NDArray], 
# #     lambda arr, p, cycle: smart_array2string(arr)
# # )

# # Register the custom formatter
# ip.display_formatter.formatters['text/plain'].for_type(list, format_list_of_ndarrays)
# ip.display_formatter.formatters['text/plain'].type_printers[np.ndarray]


# ip.display_formatter.formatters['text/plain'].type_printers.pop(list, None)


def dataframe_show_more_button(ip: "ipykernel.zmqshell.ZMQInteractiveShell") -> "ipykernel.zmqshell.ZMQInteractiveShell":
    """Adds a functioning 'show more' button below each displayed dataframe to show more rows.

    Usage:
        ip = get_ipython()
        ip = dataframe_show_more_button(ip=ip)
    """
    def _subfn_dataframe_show_more(df, initial_rows=10, default_more_rows=50):
        """Generate an HTML representation for a Pandas DataFrame with a 'show more' button."""
        total_rows = df.shape[0]
        if total_rows <= initial_rows:
            return df.to_html()

        # Create the initial view
        initial_view = df.head(initial_rows).to_html()

        # Escape backticks and newlines in the DataFrame HTML to ensure proper JavaScript string
        df_html = df.to_html().replace("`", "\\`").replace("\n", "\\n")

        # Generate the script for the 'show more' button with input for number of rows
        script = f"""
        <script type="text/javascript">
            function showMore() {{
                var numRows = document.getElementById('num-rows').value;
                if (numRows === "") {{
                    numRows = {default_more_rows};
                }} else {{
                    numRows = parseInt(numRows);
                }}
                var div = document.getElementById('dataframe-more');
                var df_html = `{df_html}`;
                var parser = new DOMParser();
                var doc = parser.parseFromString(df_html, 'text/html');
                var rows = doc.querySelectorAll('tbody tr');
                for (var i = 0; i < rows.length; i++) {{
                    if (i >= numRows) {{
                        rows[i].style.display = 'none';
                    }} else {{
                        rows[i].style.display = '';
                    }}
                }}
                div.innerHTML = doc.body.innerHTML;
            }}
        </script>
        """

        # Create the 'show more' button and input field with default value
        button_and_input = f"""
        <input type="number" id="num-rows" placeholder="Enter number of rows to display" value="{default_more_rows}">
        <button onclick="showMore()">Show more</button>
        <div id="dataframe-more"></div>
        """

        # Combine everything into the final HTML
        html = f"""
        {script}
        {initial_view}
        {button_and_input}
        """
        return HTML(html)


    ip.display_formatter.formatters['text/html'].for_type(pd.DataFrame, lambda df: display(_subfn_dataframe_show_more(df)))
    return ip




    
