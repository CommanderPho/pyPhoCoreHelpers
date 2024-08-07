"""
This type stub file was generated by pyright.
"""

from attrs import define
from pathlib import Path
from typing import Callable, Dict, List
from datetime import datetime

@define(slots=False, eq=False)
class CellExecution:
    start_time: datetime = ...
    end_time: datetime = ...
    duration: float = ...
    status: str = ...
    output: str = ...
    def to_dict(self): # -> Dict[str, Any]:
        ...
    


@define(slots=False, eq=False)
class CellInfo:
    created_at: datetime = ...
    modified_at: datetime = ...
    code: str = ...
    executions: List[CellExecution] = ...
    def to_dict(self): # -> Dict[str, Any]:
        ...
    


@define(slots=False, eq=False)
class NotebookCellExecutionLogger:
    """ Logs jupyter notebook execution activity and history
    
    Usage:
        import IPython
        from pyphocorehelpers.programming_helpers import IPythonHelpers
        from pyphocorehelpers.notebook_helpers import NotebookCellExecutionLogger

        _notebook_path:Path = Path(IPythonHelpers.try_find_notebook_filepath(IPython.extract_module_locals())).resolve() # Finds the path of THIS notebook
        _notebook_execution_logger: NotebookCellExecutionLogger = NotebookCellExecutionLogger(notebook_path=_notebook_path, enable_logging_to_file=True) # Builds a logger that records info about this notebook

    """
    notebook_path: Path = ...
    enable_logging_to_file: bool = ...
    debug_print: bool = ...
    log_file: Path = ...
    cell_info: Dict[str, CellInfo] = ...
    use_logging_subdirectory: bool = ...
    _callback_references: Dict[str, List[Callable]] = ...
    def __attrs_post_init__(self): # -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def rebuild_logging_file_info(self) -> bool:
        """ called when options change to recompute the logging output variables:

        Updates:
            self.log_file

        """
        ...
    
    def update_log_file(self): # -> None:
        ...
    
    def register_callbacks(self): # -> None:
        """ main start function """
        ...
    
    def unregister_callbacks(self): # -> None:
        """ unregister all ipynb callbacks 
        """
        ...
    


