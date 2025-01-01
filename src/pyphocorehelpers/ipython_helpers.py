from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
from IPython import get_ipython
from IPython.display import display
import numpy as np
import pandas as pd

from pyphocorehelpers.print_helpers import render_scrollable_colored_table_from_dataframe

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
        
        """
        # Execute the cell and capture the result
        result = self.shell.run_cell(cell).result
        if isinstance(result, pd.DataFrame):
            # Apply custom formatter
            display(render_scrollable_colored_table_from_dataframe(result))
        else:
            display(result)