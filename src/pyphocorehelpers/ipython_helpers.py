from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from IPython import get_ipython
from IPython.display import display
import numpy as np
import pandas as pd
import argparse

from pyphocorehelpers.print_helpers import render_scrollable_colored_table_from_dataframe, print_keys_if_possible

# ==================================================================================================================== #
# 2024-08-20 Refactored to new `"pho-jupyter-preview-widget"` library                                                  #
# ==================================================================================================================== #


@magics_class
class CustomFormatterMagics(Magics):
    """ 
    from pyphocorehelpers.ipython_helpers import CustomFormatterMagics
    
    # Register the magic
	get_ipython().register_magics(CustomFormatterMagics)

    """
    @cell_magic
    def scrollable_colored_table(self, line, cell):
        """ usage:
        
        %%scrollable_colored_table
        
        compatable with `InteractiveShell.ast_node_interactivity = "all"` and handles multiple outputs gracefully.
        
        """

        # Split the magic line by commas to get individual key-value pairs (like `%%ndarray_preview height=500, width=200, include_plaintext_repr=False`)
        # params = line.split(',')
        # config = _parse_ndarray_preview_params(line=line) 
        
        ip = get_ipython()
        
        ## Backup the current NDArray formatter
        _bak_formatter = ip.display_formatter.formatters['text/html'].type_printers.pop(pd.DataFrame, None)
            
        # Register the custom display function for pd.DataFrames for the duration of the cell:
        ip.display_formatter.formatters['text/html'].for_type(pd.DataFrame, lambda df: render_scrollable_colored_table_from_dataframe(df, output_fn=str, max_rows_to_render_for_performance=90)) # , height=height, width=width
    
        # Split the cell into individual lines
        cell_lines = cell.splitlines()
        cell_outputs = []
        
        # Execute each line and capture output (for compatibility with `InteractiveShell.ast_node_interactivity = "all"` to handle multiple outputs gracefully)
        for line in cell_lines:
            exec(line, self.shell.user_ns, self.shell.user_ns)
            
            # If the last line was an expression, capture its value to display
            if line.strip() and not line.strip().startswith('#'):
                try:
                    output = eval(line, self.shell.user_ns, self.shell.user_ns)
                    if output is not None:
                        display(output)
                        cell_outputs.append(output)
                        
                except BaseException:
                    pass  # Ignore errors for non-expressions or if exec-ed code raises an exception

        # Display the output using the custom formatter ______________________________________________________________________ #

        # Remove the custom formatter
        ip.display_formatter.formatters['text/html'].type_printers.pop(pd.DataFrame, None)

        ## Restore the previous formatter
        if _bak_formatter is not None:
            ip.display_formatter.formatters['text/html'].for_type(pd.DataFrame, _bak_formatter)
        

    @line_magic
    def keys(self, line):
        """ prints the passed object's keys hieararchically using `print_keys_if_possible`. Takes an optional -d for --depth passed as `max_depth=depth`

        Usage:

        Example 1:
            %keys trial_by_trial_window.plots_data

        Example 2 (custom `max_depth` from `--depth`):
            %keys trial_by_trial_window.plots_data --depth 2

        Example 3 (custom `max_depth` from `-d`):
            %keys trial_by_trial_window.plots_data -d 2

        """
        parser = argparse.ArgumentParser()
        parser.add_argument("name")
        parser.add_argument("--depth", "-d", type=int, default=2)

        args = parser.parse_args(line.split())

        # obj = eval(args.name) ## this does not work because it's being evaluated in the wrong namespace.
        obj = eval(args.name, self.shell.user_ns, self.shell.user_ns) ## evaluate in the calling namespace

        print_keys_if_possible(args.name, obj, max_depth=args.depth)


    # @line_magic
    # def keys(line):
    #     """ prints the keys """
    #     name = line.strip()
    #     obj = eval(name)
    #     print_keys_if_possible(name, obj, max_depth=2)
