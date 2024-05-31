"""

You can use `pd.describe_option()` to see the various customization options you can set for pandas

"""
import numpy as np
import pandas as pd
import ipykernel # ip: "ipykernel.zmqshell.ZMQInteractiveShell)" = IPython.get_ipython()




def set_pho_preferences(is_concise=False):
    """ Sets Pho Hale's general preferences for Jupyter Notebooks
    
    Includes increasing the number of rows/columns displayed for Pandas dataframes, and setting the numpy print options to be full-width for the jupyter notebook.
    """
    set_pho_numpy_display_preferences(is_concise=is_concise)
    set_pho_pandas_display_preferences(is_concise=is_concise)
    
def set_pho_preferences_verbose():
    is_concise=False
    set_pho_numpy_display_preferences(is_concise=is_concise)
    set_pho_pandas_display_preferences(is_concise=is_concise)

    
def set_pho_preferences_concise():
    is_concise=True
    set_pho_numpy_display_preferences(is_concise=is_concise)
    set_pho_pandas_display_preferences(is_concise=is_concise)
    

    
def set_pho_pandas_display_preferences(is_concise=True):
    import pandas as pd
    ## Pandas display options
    # pd.set_option('display.max_columns', None)  # or 1000
    # pd.set_option('display.max_rows', None)  # or 1000
    # pd.set_option('display.max_colwidth', -1)  # or 199
    pd.set_option('display.width', 1000)
    # pd.set_option('display.max_columns', None)  # or 1000
    # pd.set_option('display.max_rows', None)  # or 1000
    # pd.set_option('display.max_colwidth', -1)  # or 199
    # pd.set_option('display.width', 1000)
    pd.set_option('display.show_dimensions', True) # always shows dimensions even if the full dataframe can be printed.
    pd.set_option('display.max_columns', 256) # maximum number of columns to display
    if not is_concise:
        pd.set_option('display.min_rows', 30)
        pd.set_option('display.max_rows', 50)
    else:
        pd.set_option('display.min_rows', None)
        pd.set_option('display.max_rows', 12)

def set_pho_numpy_display_preferences(is_concise=True):
    import numpy as np
    ## Numpy display options:
    # Set up numpy print options to only wrap at window width:
    if is_concise:
        edgeitems = None
    else:
        edgeitems = 30
        
    np.set_printoptions(edgeitems=edgeitems, linewidth=100000, formatter=dict(float=lambda x: "%g" % x))
    # np.set_printoptions(edgeitems=3, linewidth=4096, formatter=dict(float=lambda x: "%.3g" % x))
    np.core.arrayprint._line_width = 144
    


# ---------------------------------------------------------------------------- #
#                       Jupyter Datatype Printing Helpers                      #
# ---------------------------------------------------------------------------- #

def array_repr_with_graphical_shape(ip: "ipykernel.zmqshell.ZMQInteractiveShell") -> "ipykernel.zmqshell.ZMQInteractiveShell":
    """Generate an HTML representation for a NumPy array, similar to Dask.
        
    from preferences_helpers import array_graphical_shape
    from pyphocorehelpers.print_helpers import array_preview_with_graphical_shape_repr_html
    
    # Register the custom display function for NumPy arrays
    import IPython
    
    ip: "ipykernel.zmqshell.ZMQInteractiveShell)" = IPython.get_ipython()


    """
    from pyphocorehelpers.print_helpers import array_preview_with_graphical_shape_repr_html
    # Register the custom display function for NumPy arrays
    ip.display_formatter.formatters['text/html'].for_type(np.ndarray, lambda arr: array_preview_with_graphical_shape_repr_html(arr))
    return ip

    
def dataframe_show_more_button(ip: "ipykernel.zmqshell.ZMQInteractiveShell") -> "ipykernel.zmqshell.ZMQInteractiveShell":
    """Adds a functioning "show more" button below each displayed dataframe to show more rows
            
    Usage:
        from pyphocorehelpers.preferences_helpers import array_repr_with_graphical_shape, dataframe_show_more_button

        ip = get_ipython()

        ip = array_repr_with_graphical_shape(ip=ip)
        ip = dataframe_show_more_button(ip=ip)


    """
    # Register the custom display function for NumPy arrays
    # ip.display_formatter.formatters['text/html'].for_type(pd.DataFrame, lambda df: ????(df))

    def show_more(df, show_rows=5, _id=None):
        # Generate a default id based on the object id if not specified
        _id = f"df-{id(df)}" if _id is None else _id
        return f"""
        <div id="{_id}" class="dataframe-container">
            {df.head(show_rows).to_html()}
            <button onclick="showMoreRows('{_id}', {show_rows})">Show More</button>
        </div>
        <script>
        function showMoreRows(id, showRows) {{
            var dfContainer = document.getElementById(id);
            var currentRows = dfContainer.getElementsByTagName('table')[0].rows.length - 1; // Subtract 1 for the header
            var totalRows = {len(df)};
            var newRows = Math.min(totalRows, currentRows + showRows);
            var xhr = new XMLHttpRequest();
            xhr.open('GET', `/_show_more?id=${id}&rows=${newRows}`, false); // Synchronous request for simplicity
            xhr.send();
            if (xhr.status === 200) {{
                dfContainer.innerHTML = xhr.responseText + dfContainer.innerHTML;
            }}
        }}
        </script>
        """

    # Register the custom display function for pandas DataFrames
    ip.display_formatter.formatters['text/html'].for_type(
        pd.DataFrame, lambda df, show_rows=5, unique_id=None:
        show_more(df, show_rows, unique_id)
    )

    return ip

    
