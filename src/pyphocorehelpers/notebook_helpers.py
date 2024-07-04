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

@attr.s
class CellExecution:
    start_time = attr.ib(factory=datetime.now)
    end_time = attr.ib(default=None)
    duration = attr.ib(default=None)
    status = attr.ib(default="running")

@attr.s
class CellInfo:
    created_at = attr.ib(factory=datetime.now)
    modified_at = attr.ib(default=None)
    code = attr.ib(default="")
    executions = attr.ib(factory=list)


@define(slots=False, eq=False)
class NotebookCellExecutionLogger:
    """ Logs jupyter notebook execution activity and history
    
    Usage:
        from pyphocorehelpers.programming_helpers import IPythonHelpers
        from pyphocorehelpers.notebook_helpers import NotebookCellExecutionLogger

        _notebook_path:Path = Path(IPythonHelpers.try_find_notebook_filepath(IPython.extract_module_locals())).resolve() # Finds the path of THIS notebook
        _notebook_execution_logger: NotebookCellExecutionLogger = NotebookCellExecutionLogger(notebook_path=_notebook_path, enable_logging_to_file=True) # Builds a logger that records info about this notebook

    """
    notebook_path: Path = field()
    enable_logging_to_file: bool = field(default=True)
    log_file: Path = field(default=None, init=False)
    cell_info: Dict[str, CellInfo] = field(default=Factory(dict))

    def __attrs_post_init__(self):
        assert self.notebook_path.exists()
        notebook_directory = self.notebook_path.parent.resolve()
        notebook_filename: str = self.notebook_path.stem
        log_filename = f"_cell_execution_log_{notebook_filename}.json"
        log_filename: str = sanitize_filename_for_Windows(log_filename)
        self.log_file = notebook_directory.joinpath(log_filename)
        self.record_execution_details()

    
    def update_log_file(self):
        if self.enable_logging_to_file:
            with open(self.log_file, 'w') as f:
                cell_info_dict = {k:attr.asdict(v, recurse=True) for k, v in self.cell_info.items()}
                json.dump(cell_info_dict, f, default=str, indent=4)


    def record_execution_details(self):
        """ main start function """
        ip = get_ipython()

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

        ip.events.register('pre_run_cell', pre_run_cell)
        ip.events.register('post_run_cell', post_run_cell)
        ip.events.register('pre_run_cell', cell_create_or_modify)




