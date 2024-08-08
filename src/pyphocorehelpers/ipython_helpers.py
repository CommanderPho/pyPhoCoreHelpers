from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from IPython import get_ipython
from IPython.display import display
import numpy as np

@magics_class
class MyMagics(Magics):
    """ 
    from pyphocorehelpers.ipython_helpers import MyMagics
    
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

        """ Allows the user to change the thumbnail image size like:
    
        # call the line magic
        %config_ndarray_preview width=500 
        
        
        """
        from pyphocorehelpers.preferences_helpers import array_repr_with_graphical_preview
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()

        width = None
        height = None
        if key == "width":
            width = int(value)
        elif key == "height":
            height = int(value)
        else:
            raise KeyError(f"Unknown configuration key: {key}")

        ip = get_ipython()
        array_repr_with_graphical_preview(ip=ip, width=width, height=height)
        

    @cell_magic
    def ndarray_preview(self, line, cell):
        """ 
        %%ndarray_preview height=500, width=200, include_plaintext_repr=False
        
        %%ndarray_preview height=None, width=100, include_plaintext_repr=True, include_shape=False, horizontal_layout=False
        
        """
        from pyphocorehelpers.preferences_helpers import array_repr_with_graphical_preview
        from pyphocorehelpers.print_helpers import array_preview_with_heatmap_repr_html
            
        debug_print = True

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
            
        if debug_print:
            print(f'config: {config}\n')
            
        # integer_keys = ['width', 'height']
        # for k in integer_keys:
        #     if k in config:
        #         if config[k] is not None:
        #             config[k] = int(config[k]) # convert to integer
        
            
        ip = get_ipython()
        
        ## Backup the current NDArray formatter
        _bak_formatter = ip.display_formatter.formatters['text/html'].type_printers.pop(np.ndarray, None)
        # if _bak_formatter is not None:
        #     print(_bak_formatter)
            
        # Register the custom display function for NumPy arrays for the duration of the cell:
        # ip.display_formatter.formatters['text/html'].for_type(np.ndarray, lambda arr: array_preview_with_heatmap_repr_html(arr, **config))
        array_repr_with_graphical_preview(ip=ip, **config)
        
        # Execute the cell content and capture the output
        exec(cell, self.shell.user_ns)
        output = eval(cell, self.shell.user_ns)

        # Display the output using the custom formatter ______________________________________________________________________ #

        # Fetch the variables created in the cell for display
        # output = {var_name: self.shell.user_ns[var_name] for var_name in self.shell.user_ns if isinstance(self.shell.user_ns[var_name], np.ndarray)}
        display(output)

        # for var in output.values():
        #     display(var)

        # display(output)
        # return output
        
        # Remove the custom formatter
        ip.display_formatter.formatters['text/html'].type_printers.pop(np.ndarray, None)

        ## Restore the previous formatter
        if _bak_formatter is not None:
            ip.display_formatter.formatters['text/html'].for_type(np.ndarray, _bak_formatter)
            
        # Return the output to display it in the cell
        # return output
    

# ## Usage:
# # Register the magic
# ip = get_ipython()
# ip.register_magics(MyMagics)
