"""
This type stub file was generated by pyright.
"""

from IPython.core.magic import Magics, cell_magic, magics_class

@magics_class
class CustomFormatterMagics(Magics):
    """ 
    from pyphocorehelpers.ipython_helpers import CustomFormatterMagics
    
    # Register the magic
	get_ipython().register_magics(CustomFormatterMagics)

    """
    @cell_magic
    def scrollable_colored_table(self, line, cell): # -> None:
        """ usage:
        
        %%scrollable_colored_table
        
        compatable with `InteractiveShell.ast_node_interactivity = "all"` and handles multiple outputs gracefully.
        
        """
        ...
    


