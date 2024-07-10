import attr
from attrs import define, field, Factory
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from IPython.core.magics.execution import _format_time
from datetime import datetime
import json
from pyphocorehelpers.Filesystem.path_helpers import sanitize_filename_for_Windows
from ipywidgets import get_ipython # required for IPythonHelpers.cell_vars

def _safely_remove_all_module_registered_callbacks():
    """ safely remove all callbacks registered by my custom module even if we lost references to them
    """
    ip = get_ipython()
    # print(f"ip.events.callbacks: {ip.events.callbacks}")
    # ip.events.callbacks:
        # {'pre_execute': [<bound method InteractiveShell._clear_warning_registry of <ipykernel.zmqshell.ZMQInteractiveShell object at 0x0000021DBB8B56D0>>],
        # 'pre_run_cell': [<bound method AutoreloadMagics.pre_run_cell of <IPython.extensions.autoreload.AutoreloadMagics object at 0x0000021DBBA59940>>,
        # <function pyphocorehelpers.notebook_helpers.NotebookCellExecutionLogger.record_execution_details.<locals>.pre_run_cell(info)>,
        # <function pyphocorehelpers.notebook_helpers.NotebookCellExecutionLogger.record_execution_details.<locals>.cell_create_or_modify(info)>],
        # 'post_execute': [<bound method AutoreloadMagics.post_execute_hook of <IPython.extensions.autoreload.AutoreloadMagics object at 0x0000021DBBA59940>>,
        # <function matplotlib.pyplot._draw_all_if_interactive() -> 'None'>],
        # 'post_run_cell': [<function pyphocorehelpers.notebook_helpers.NotebookCellExecutionLogger.record_execution_details.<locals>.post_run_cell(result)>],
        # 'shell_initialized': []}
    for a_callback_type, cb_list in ip.events.callbacks.items():
        for a_fn in cb_list:
            if a_fn.__module__ in ['pyphocorehelpers.notebook_helpers']:
                ip.events.unregister(a_callback_type, a_fn)


@define(slots=False, eq=False)
class CellExecution:
    start_time: datetime = field(factory=datetime.now)
    end_time: datetime = field(default=None)
    duration: float = field(default=None)
    status: str = field(default="running")
    output: str = field(default="", metadata={'notes': 'Added field to store output history'})  # Added field to store output history

    def to_dict(self):
        return attr.asdict(self, recurse=True)
        # return attr.asdict(self, recurse=True, value_serializer=self.serializer)

    # @staticmethod
    # def serializer(value, attr, obj, is_key):
    #     if isinstance(value, datetime):
    #         return value.isoformat()
    #     if isinstance(value, (list, dict)):
    #         return value
    #     return str(value)
    


@define(slots=False, eq=False)
class CellInfo:
    created_at: datetime = field(factory=datetime.now)
    modified_at: datetime = field(default=None)
    code: str = field(default="")
    executions: List[CellExecution] = field(factory=list)

    def to_dict(self):
        return attr.asdict(self, recurse=True) # , value_serializer=self.serializer

    # @staticmethod
    # def serializer(value, attr, obj, is_key):
    #     if isinstance(value, datetime):
    #         return value.isoformat()
    #     if isinstance(value, (list, dict)):
    #         return value
    #     return str(value)
    


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
    notebook_path: Path = field()
    enable_logging_to_file: bool = field(default=True)
    debug_print: bool = field(default=False)
    log_file: Path = field(default=None, init=False)
    cell_info: Dict[str, CellInfo] = field(default=Factory(dict))
    use_logging_subdirectory: bool = field(default=True, metadata={'notes': "if true, logs to the notebook's parent folder/._cell_execution_logs subdirectory"}) # 
    _callback_references: Dict[str, List[Callable]] = field(default=Factory(dict))

    def __attrs_post_init__(self):
        _did_logging_file_change: bool = self.rebuild_logging_file_info()
        self.register_callbacks()

    def __del__(self):
        self.unregister_callbacks()



    def rebuild_logging_file_info(self) -> bool:
        """ called when options change to recompute the logging output variables:

        Updates:
            self.log_file

        """
        assert self.notebook_path.exists()
        notebook_directory = self.notebook_path.parent.resolve()
        notebook_filename: str = self.notebook_path.stem
        log_filename = f"_cell_execution_log_{notebook_filename}.json"
        log_filename: str = sanitize_filename_for_Windows(log_filename)
        
        if self.use_logging_subdirectory:
            log_directory = notebook_directory.joinpath('._cell_execution_logs').resolve()
            log_directory.mkdir(exist_ok=True)
        else:
            log_directory = notebook_directory

        new_log_File_directory = log_directory.joinpath(log_filename).resolve()

        did_logging_file_change: bool = ((self.log_file is None) or (self.log_file != new_log_File_directory))
        self.log_file = log_directory.joinpath(log_filename) # update the logging file
        return did_logging_file_change


    
    def update_log_file(self):
        if self.enable_logging_to_file:
            with open(self.log_file, 'w') as f:
                # cell_info_dict = {k:attr.asdict(v, recurse=True) for k, v in self.cell_info.items()}
                cell_info_dict = {k: v.to_dict() for k, v in self.cell_info.items()}
                json.dump(cell_info_dict, f, default=str, indent=4)



    def register_callbacks(self):
        """ main start function """
        from ipywidgets import get_ipython # required for IPythonHelpers.cell_vars

        ip = get_ipython()
        self.unregister_callbacks() ## unregister existing callbacks

        def pre_run_cell(info):
            cell_id = info.cell_id
            if cell_id not in self.cell_info:
                self.cell_info[cell_id] = CellInfo(code=info.raw_cell)
            self.cell_info[cell_id].executions.append(CellExecution())
            self.update_log_file()

        def post_run_cell(result):
            cell_id = result.info.cell_id
            if cell_id in self.cell_info and self.cell_info[cell_id].executions:
                exec_info = self.cell_info[cell_id].executions[-1]
                exec_info.end_time = datetime.now()
                exec_info.duration = (exec_info.end_time - exec_info.start_time).total_seconds()
                exec_info.status = "failed" if result.error_in_exec else "success"
                if result.result is None:
                    exec_info.output = ''
                else:
                    if isinstance(result.result, str):
                        exec_info.output = result.result # Capture the output. A list of things
                    else:
                        # might be a list/tuple of strings
                        exec_info.output = '_________________________________________________________________ \n'.join([str(v) for v in result.result]) # combine multiple outputs into a single cell
                    if self.debug_print:
                        print(f"Cell executed in: {_format_time(exec_info.duration)}")

                self.update_log_file()

        def cell_create_or_modify(info):
            cell_id = info.cell_id
            if cell_id in self.cell_info:
                self.cell_info[cell_id].modified_at = datetime.now()
                self.cell_info[cell_id].code = info.raw_cell
            else:
                self.cell_info[cell_id] = CellInfo(code=info.raw_cell)
            self.update_log_file()

        # ip.events.register('pre_run_cell', pre_run_cell)
        # ip.events.register('post_run_cell', post_run_cell)
        # ip.events.register('pre_run_cell', cell_create_or_modify)

        ## hardcoded
        self._callback_references = {'pre_run_cell': [pre_run_cell, cell_create_or_modify],
            'post_run_cell': [post_run_cell],
        }

        for a_callback_type, a_callback_fns_list in self._callback_references.items():
            # a_callback_type: like 'pre_run_cell'
            for a_fn in a_callback_fns_list:
                ip.events.register(a_callback_type, a_fn)


    def unregister_callbacks(self):
        """ unregister all ipynb callbacks 
        """
        ip = get_ipython()
        if ip:
            try:
                for a_callback_type, a_callback_fns_list in self._callback_references.items():
                    # a_callback_type: like 'pre_run_cell'
                    for a_fn in a_callback_fns_list:
                        ip.events.unregister(a_callback_type, a_fn)
            except BaseException as err:
                print(f'could not remove all callbacks due to {err}')
            finally:
                ## ensure that all are removed no matter what:
                _safely_remove_all_module_registered_callbacks
                self._callback_references = {} # empty

