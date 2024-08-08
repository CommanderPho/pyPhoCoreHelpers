from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from IPython import get_ipython
from IPython.display import display
import numpy as np


def _parse_ndarray_preview_params(line: str) -> Dict:
    """ 
    %%ndarray_preview height=500, width=200, include_plaintext_repr=False
    
    %%ndarray_preview height=None, width=100, include_plaintext_repr=True, include_shape=False, horizontal_layout=False
    
    """
    # Split the magic line by commas to get individual key-value pairs (like `%%ndarray_preview height=500, width=200, include_plaintext_repr=False`)
    params = line.split(',')
    config = {}
    
    for param in params:
        key, value = param.split('=')
        key = key.strip()
        value = value.strip()
        # Convert string representation to appropriate type
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.lower() =='none':
            value = None
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        config[key] = value
        
    return config


@magics_class
class MyMagics(Magics):
    """ 
    ## Usage:

        from pyphocorehelpers.ipython_helpers import MyMagics
        # Register the magic
        ip = get_ipython()
        ip.register_magics(MyMagics)

    """
    @cell_magic
    def setvar(self, line, cell):
        # Execute the cell content and set a variable in the user's namespace
        exec(cell, self.shell.user_ns)
        # Optionally, you can add any additional logic here
        # For example, setting a specific variable
        var_name = line.strip()
        if var_name:
            self.shell.user_ns[var_name] = eval(var_name, self.shell.user_ns)
        
    @line_magic
    def config_ndarray_preview(self, line):
        """ Updates the display preview config (persistantly). Is a line magic because it updates the ipython display_formatter directly and will persist for the remainder (and the following) cells
            Allows the user to change the thumbnail image size like:
    
        # call the line magic
        %config_ndarray_preview width=500 
        
        """
        from pyphocorehelpers.preferences_helpers import array_repr_with_graphical_preview
        config = _parse_ndarray_preview_params(line=line)

        ip = get_ipython()
        array_repr_with_graphical_preview(ip=ip, **config)
        

    @cell_magic
    def ndarray_preview(self, line, cell):
        """ 
        %%ndarray_preview height=500, width=200, include_plaintext_repr=False
        
        %%ndarray_preview height=None, width=100, include_plaintext_repr=True, include_shape=False, horizontal_layout=False
        
        """
        from pyphocorehelpers.preferences_helpers import array_repr_with_graphical_preview
            
        debug_print = False

        # Split the magic line by commas to get individual key-value pairs (like `%%ndarray_preview height=500, width=200, include_plaintext_repr=False`)
        params = line.split(',')
        config = _parse_ndarray_preview_params(line=line) 
        
        if debug_print:
            print(f'config: {config}\n')
            
        ip = get_ipython()
        
        ## Backup the current NDArray formatter
        _bak_formatter = ip.display_formatter.formatters['text/html'].type_printers.pop(np.ndarray, None)
            
        # Register the custom display function for NumPy arrays for the duration of the cell:
        array_repr_with_graphical_preview(ip=ip, **config)
        
        # Execute the cell content and capture the output
        exec(cell, self.shell.user_ns)
        output = eval(cell, self.shell.user_ns)

        # Display the output using the custom formatter ______________________________________________________________________ #

        # Fetch the variables created in the cell for display
        # output = {var_name: self.shell.user_ns[var_name] for var_name in self.shell.user_ns if isinstance(self.shell.user_ns[var_name], np.ndarray)}
        display(output)
        
        # Remove the custom formatter
        ip.display_formatter.formatters['text/html'].type_printers.pop(np.ndarray, None)

        ## Restore the previous formatter
        if _bak_formatter is not None:
            ip.display_formatter.formatters['text/html'].for_type(np.ndarray, _bak_formatter)
            
        # Return the output to display it in the cell
        # return output
    


