


from operator import is_


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
    if not is_concise:
        pd.set_option('display.min_rows', 30)
        pd.set_option('display.max_rows', 50)
    else:
        pd.set_option('display.min_rows', None)
        pd.set_option('display.max_rows', 30)

    
    
    
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
    

# def set_pho_jupyter_lab_display_preferences():
#     %config Completer.use_jedi = False