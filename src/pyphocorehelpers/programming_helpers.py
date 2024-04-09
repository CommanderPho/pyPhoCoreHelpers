# title: programming_helpers.py
# date: 2023-05-08 14:21:48
# purpose: Created to support programming and consolidation of programming-related helpers into a single location. Previously all were scattered around the various other helpers.
import os
import sys
import contextlib
from collections import namedtuple
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Tuple, Any
from functools import wraps
import numpy as np
import pandas as pd
import inspect # for IPythonHelpers
from enum import Enum, unique
import re
import ast
import nbformat
import IPython
from IPython.display import display, Javascript
import json
from attrs import define, field, Factory
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum
from pyphocorehelpers.function_helpers import function_attributes

# from pyphocorehelpers.function_helpers import function_attributes, _custom_function_metadata_attribute_names

# ==================================================================================================================== #
# Documentation Decorators                                                                                             #
# ==================================================================================================================== #

def documentation_tags(func, *tags):
    """Adds documentations tags to a function or class

    Usage:
        from pyphocorehelpers.programming_helpers import documentation_tags
        @documentation_tags('tag1', 'tag2')
        def my_function():
            ...

        Access via:
            my_function.tags

    """
    @wraps(func)
    def decorator(func):
        func.tags = tags
        return func
    return decorator


# ==================================================================================================================== #
# Function Attributes Decorators                                                                                       #
# ==================================================================================================================== #
_custom_metadata_attribute_names = dict(short_name=None, tags=None, creation_date=None,
                                         input_requires=None, output_provides=None,
                                         uses=None, used_by=None,
                                         related_items=None, # references to items related to this definition
                                         pyqt_signals_emitted=None # QtCore.pyqtSignal()s emitted by this object
)


def metadata_attributes(short_name=None, tags=None, creation_date=None, input_requires=None, output_provides=None, uses=None, used_by=None, related_items=None,  pyqt_signals_emitted=None):
    """Adds generic metadata attributes to a function or class
    Aims to generalize `pyphocorehelpers.function_helpers.function_attributes`

    ```python
        from pyphocorehelpers.programming_helpers import metadata_attributes

        @metadata_attributes(short_name='pf_dt_sequential_surprise', tags=['tag1','tag2'], input_requires=[], output_provides=[], uses=[], used_by=[])
        def _perform_time_dependent_pf_sequential_surprise_computation(computation_result, debug_print=False):
            # function body
    ```

    func.short_name, func.tags, func.creation_date, func.input_requires, func.output_provides, func.uses, func.used_by
    """
    # decorator = function_attributes(func) # get the decorator provided by function_attributes
    def decorator(func):
        func.short_name = short_name
        func.tags = tags
        func.creation_date = creation_date
        func.input_requires = input_requires
        func.output_provides = output_provides
        func.uses = uses
        func.used_by = used_by
        func.related_items = related_items
        func.pyqt_signals_emitted = pyqt_signals_emitted
        return func
    return decorator


# ==================================================================================================================== #
# Function and Class Metadata Accessors                                                                                #
# ==================================================================================================================== #


def build_metadata_property_reverse_search_map(a_fn_dict, a_metadata_property_name='short_name'):
    """allows lookup of key into the original dict via a specific value of a specified property
    Usage:
        from pyphocorehelpers.programming_helpers import build_metadata_property_reverse_search_map
        short_name_reverse_lookup_map = build_metadata_property_reverse_search_map(curr_active_pipeline.registered_merged_computation_function_dict, a_metadata_property_name='short_name')
        short_name_search_value = 'long_short_fr_indicies_analyses'
        short_name_reverse_lookup_map[short_name_search_value] # '_perform_long_short_firing_rate_analyses'
    """
    metadata_property_reverse_search_map = {getattr(a_fn, a_metadata_property_name, None):a_name for a_name, a_fn in a_fn_dict.items() if getattr(a_fn, a_metadata_property_name, None)}
    return metadata_property_reverse_search_map


def build_fn_properties_dict(a_fn_dict, included_attribute_names_list:Optional[List]=None, private_attribute_names_list:List[str]=['__name__', '__doc__'], debug_print:bool=False) -> Dict:
    """ Given a dictionary of functions tries to extract the metadata

    from pyphocorehelpers.programming_helpers import build_fn_properties_dict

    Example: Merged Functions:
        computation_fn_dict = curr_active_pipeline.registered_merged_computation_function_dict
        computation_fn_metadata_dict = build_fn_properties_dict(curr_active_pipeline.registered_merged_computation_function_dict, ['__name__', 'short_name'], private_attribute_names_list=[])
        computation_fn_metadata_dict
    """
    data_dict = {}
    for a_name, a_fn in a_fn_dict.items():
        if debug_print:
            print(f'a_name: {a_name}')
            # , '__annotations__', '__class__', '__closure__', '__code__', '__module__', '__qualname__', '__sizeof__', '__str__', '__subclasshook__']
        if included_attribute_names_list is None:
            all_public_fn_attribute_names = [a_name for a_name in dir(a_fn) if (not a_name.startswith('_'))] # enumerate all non-private (starting with a '_' character) members
        else:
            # include only the items
            all_public_fn_attribute_names = [a_name for a_name in dir(a_fn) if (not a_name.startswith('_')) and (a_name in included_attribute_names_list)] # enumerate all non-private (starting with a '_' character) members

        all_fn_attribute_names = all_public_fn_attribute_names + private_attribute_names_list
        all_fn_attribute_values = [a_fn.__getattribute__(a_name) for a_name in all_fn_attribute_names]
        if debug_print:
            print(f'\t{all_fn_attribute_names}')
            print(f'\t{all_fn_attribute_values}')

        a_fn_metadata_dict = dict(zip(all_fn_attribute_names, all_fn_attribute_values))
        data_dict[a_name] = a_fn_metadata_dict

    return data_dict


def build_fn_properties_df(a_fn_dict, included_attribute_names_list:Optional[List]=None, private_attribute_names_list:List[str]=['__name__', '__doc__'], debug_print:bool=False) -> pd.DataFrame:
    """ Given a dictionary of functions tries to extract the metadata

    from pyphocorehelpers.programming_helpers import build_fn_properties_df

    Usage Examples:
        display_fn_dict = curr_active_pipeline.registered_display_function_dict
        display_fn_df = build_fn_properties_df(display_fn_dict)
        display_fn_df

    Example 2: Global Computation Functions
        global_computation_fn_dict = curr_active_pipeline.registered_global_computation_function_dict
        global_computation_fn_df = build_fn_properties_df(global_computation_fn_dict)
        global_computation_fn_df

    Example 3: Merged Functions:
        computation_fn_dict = curr_active_pipeline.registered_merged_computation_function_dict
        computation_fn_df = build_fn_properties_df(computation_fn_dict)
        computation_fn_df
    """
    data = []
    for a_name, a_fn in a_fn_dict.items():
        # print(f'\t{dir(a_fn)}') # ['__annotations__', '__call__', '__class__', '__closure__', '__code__', '__defaults__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__get__', '__getattribute__', '__globals__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__kwdefaults__', '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__', '__qualname__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'conforms_to', 'creation_date', 'input_requires', 'is_global', 'output_provides', 'related_items', 'short_name', 'tags', 'used_by', 'uses', 'validate_computation_test']
        # ['conforms_to', 'creation_date', 'input_requires', 'is_global', 'output_provides', 'related_items', 'short_name', 'tags', 'used_by', 'uses', 'validate_computation_test']
        if debug_print:
            print(f'a_name: {a_name}')
        if included_attribute_names_list is None:
            all_public_fn_attribute_names = [a_name for a_name in dir(a_fn) if (not a_name.startswith('_'))] # enumerate all non-private (starting with a '_' character) members
        else:
            # include only the items
            all_public_fn_attribute_names = [a_name for a_name in dir(a_fn) if (not a_name.startswith('_')) and (a_name in included_attribute_names_list)] # enumerate all non-private (starting with a '_' character) members
        all_fn_attribute_names = all_public_fn_attribute_names + private_attribute_names_list
        all_fn_attribute_values = [a_fn.__getattribute__(a_name) for a_name in all_fn_attribute_names]
        if debug_print:
            print(f'\t{all_fn_attribute_names}')
            print(f'\t{all_fn_attribute_values}')

        a_fn_metadata_dict = dict(zip(all_fn_attribute_names, all_fn_attribute_values))
        data.append(a_fn_metadata_dict)

    df = pd.DataFrame.from_dict(data)
    # long_name_column = df.pop('__name__')
    return df


def copy_to_clipboard(code_str: str, message_print=True):
    """ tries a clean method that uses the jaraco.clipboard library, but falls back to Pandas for compatibility if it isn't available
    """
    try:
        # Try to import jaraco.clipboard
        import jaraco.clipboard as clipboard
        clipboard.copy(code_str)

    except ImportError:
        # If jaraco.clipboard is not available, fall back to pandas method that leaves extra double quotes wrapping the clipboard string
        df = pd.DataFrame([code_str])
        df.to_clipboard(index=False,header=False)

    if message_print:
        print(f'Copied "{code_str}" to clipboard!')


def copy_image_to_clipboard(image, message_print=True):
    """ copies the passed image to the system clipboard. Only works on Windows.
    
    from PIL import Image
    from pyphocorehelpers.programming_helpers import copy_image_to_clipboard
    
     
    
    from pyphocorehelpers.programming_helpers import copy_image_to_clipboard

    canvas = self.ui.canvas

    canvas.draw()  # Ensure the canvas has been drawn once before copying the figure        
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    # Send the image to the clipboard
    copy_image_to_clipboard(img)
    buf.close()
        
     """
    import sys
    import subprocess
    import io
    from PIL import Image

    # Input is of type PIL.Image

    def _subfn_send_to_clipboard(data, clip_type='image/png'):
        if sys.platform == 'win32':
            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            data = data[14:]  # Remove the 14-byte BMP header
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()
        elif sys.platform == 'linux':
            """ Requires xclip, untested.
            sudo apt-get install xclip  # Debian/Ubuntu
            # or
            sudo yum install xclip  # CentOS/Fedora
            """
            process = subprocess.Popen(['xclip', '-selection', 'clipboard', '-t', clip_type, '-i'], stdin=subprocess.PIPE)
            process.communicate(data)
        else:
            raise NotImplementedError(f'unimplemented platform!')

        
    # Determine format based on platform
    img_format: str = 'PNG' if sys.platform == 'linux' else 'BMP'

    # Output the image to bytes
    output = io.BytesIO()
    image.convert('RGB').save(output, img_format)
    data = output.getvalue()
    output.close()

    # Send the image to the clipboard
    _subfn_send_to_clipboard(data, clip_type=f'image/{img_format.lower()}')
    if message_print:
        print(f'Copied image to clipboard!')

    
    


class CodeParsers:
    """

    from pyphocorehelpers.programming_helpers import CodeParsers

    print(extract_variable_names(CodeParsers.code_block))
    """

    @classmethod
    def extract_imported_names(cls, code):
        tree = ast.parse(code)
        imported_names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.name if node.level == 0 else f"{node.module}.{alias.name}"
                    imported_names.add(name)

        return imported_names


    @classmethod
    def extract_assigned_variable_names(cls, code: str) -> List[str]:
        """
        Extracts the names of all assigned variables from the given code block.

        Args:
        code (str): A string containing the Python code from which to extract variable names.

        Returns:
        list: A list of variable names assigned in the code.

        Usage:
            # # Example usage
            # print(extract_variable_names(code_block))

        """
        tree = ast.parse(code)
        var_names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    # Handle simple assignments
                    if isinstance(target, ast.Name):
                        var_names.add(target.id)
                    # Handle unpacked assignments (e.g., a, b = c, d)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for elem in target.elts:
                            if isinstance(elem, ast.Name):
                                var_names.add(elem.id)

        return list(var_names)

    @classmethod
    def extract_referenced_names(cls, code: str) -> List[str]:
        tree = ast.parse(code)
        names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)

        return list(names)

    @classmethod
    def find_input_variables(cls, code: str):
        assigned_names = set(cls.extract_assigned_variable_names(code))
        referenced_names = set(cls.extract_referenced_names(code))

        # Input variables are those that are referenced but not assigned within the block
        input_vars = referenced_names - assigned_names
        return list(input_vars)

    @classmethod
    def convert_to_kwarg_passthrough(cls, input_str, post_equals_space: bool = False, to_clipboard=True):
        """
        Convert a string of parameters into a keyword argument passthrough format.

        Prompt:

        Write a python function that given a string like 'single_plot_fixed_height=100.0, debug_test_max_num_slices=20, size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True' it should produce a string like 'single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=debug_test_max_num_slices, size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure'

        Args:
            input_str (str): Input string containing parameters.

        Returns:
            str: Formatted string with parameters as keyword arguments.

        Usage:

        CodeParsers.convert_to_kwarg_passthrough( 'single_plot_fixed_height=100.0, debug_test_max_num_slices=20, size=(15,15), dpi=72, constrained_layout=True, scrollable_figure=True')
        >>> 'single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=debug_test_max_num_slices, size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure'

        """
        out = []
        for kwarg_k_v_pair in input_str.split(', '):
            k, v = kwarg_k_v_pair.strip().split('=')
            k = k.strip()
            v = v.strip()

            if post_equals_space:
                equals_str = '= '
            else:
                equals_str = '='

            out.append(equals_str.join([k, k])) # note this is k=k, not k=v

        code_str = ', '.join(out)
        if to_clipboard:
            copy_to_clipboard(code_str)
        return code_str

    # @classmethod
    # def convert_to_classmethod(cls, input_str):
    #     """
    #     Convert a Python function definition to its @classmethod version.

    #     Args:
    #         input_str (str): Input string containing a Python function definition.

    #     Returns:
    #         str: Formatted string with @classmethod added.
    #     """
    #     lines = input_str.split('\n')
    #     first_line = lines[0].strip()
    #     indentation = ' ' * (len(lines[0]) - len(lines[0].lstrip()))

    #     if not first_line.startswith('def '):
    #         return "Input doesn't seem to be a valid function definition."

    #     method_name = first_line.split('(')[0][4:]
    #     new_method_signature = f"    @classmethod\n    def {method_name}(cls, {first_line.split('(')[1]}"

    #     return '\n'.join([lines[0], new_method_signature] + [line.replace(indentation, indentation + '    ') for line in lines[1:]])


class IPythonHelpers:
    """ various helpers useful in jupyter-lab notebooks and IPython

    import IPython
    from pyphocorehelpers.programming_helpers import IPythonHelpers
    notebook_name = IPythonHelpers.try_find_notebook_filepath(IPython.extract_module_locals())


    Future Explorations:
        from ipywidgets import get_ipython # required for IPythonHelpers.cell_vars
        import io
        from contextlib import redirect_stdout

        ipy = get_ipython()
        ipy

        ipy.config
        {'IPKernelApp': {'ip': '127.0.0.1',
        'stdin_port': 9018,
        'control_port': 9016,
        'hb_port': 9015,
        'shell_port': 9017,
        'transport': 'tcp',
        'iopub_port': 9019,
        'connection_file': '/home/halechr/.local/share/jupyter/runtime/kernel-v2-11469ce5F9Z6kztRl.json'},
        'Session': {'signature_scheme': 'hmac-sha256',
        'key': b'e8f1060a-87a5-4060-9275-77318885dab6'},
        'Completer': {'use_jedi': False},
        'IPCompleter': {'use_jedi': False}}

        # ipy.find_user_code
        # ipy.get_local_scope
        # ipy.inspector.class_get_help
        # ipy.get_ipython()

    """

    @classmethod
    def _subfn_helper_get_name(cls, lst=[]):
        local_vars = inspect.currentframe().f_back.f_locals.items()
        for i in local_vars:
            lst.append(i)
        return dict(lst)

    @classmethod
    def _helper_get_global_dict(cls):
        ## Globals method only works when defined in notebook.
        # g = globals()


        ## Inspect stack approach:
        # caller_frame = inspect.stack()[1]
        # caller_module = inspect.getmodule(caller_frame[0])
        # g = caller_module.__dict__
        # # return [name for name, entry in caller_module.__dict__.items() ]

        g = cls._subfn_helper_get_name()

        return g


    @classmethod
    def try_find_notebook_filepath(cls, module_locals=None) -> Optional[str]:
        """ tries multiple methods to get a Jupyter notebok's filepath
        Usage:
            MUST be called like:
            import IPython
            from pyphocorehelpers.programming_helpers import IPythonHelpers
            notebook_name = IPythonHelpers.try_find_notebook_filepath(IPython.extract_module_locals())

        """
        def _subfn_try_find_notebook_fp_using_javascript() -> Optional[str]:
            """Return the absolute path of the current Jupyter notebook, e.g., '/path/to/Notebook.ipynb'"""
            import json
            import os

            # JavaScript to get the notebook's base URL
            js = Javascript("""
            require(["base/js/namespace"], function(Jupyter) {
                Jupyter.notebook.kernel.execute("notebook_path = '" + 
                    Jupyter.notebook.notebook_path + "'");
            });
            """)
            display(js)

            # Wait for the Javascript command to complete execution
            try:
                notebook_path = globals()['notebook_path']
                return os.path.join(os.getcwd(), notebook_path)
            except KeyError:
                print("Can't find notebook path")
                return None


        def _subfn_try_find_notebook_fp_from_server():
            """
            Apparently this only works from if the Jupyter server is open (not in VSCode)

            notebook_path = _subfn_try_find_notebook_fp_from_server()
            print(notebook_path)

            """
            import requests
            import json
            import ipykernel
            from notebook.notebookapp import list_running_servers
            kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
            for server in list_running_servers():
                response = requests.get(requests.compat.urljoin(server['url'], 'api/sessions'),
                                        headers={'Authorization': 'token ' + server.get('token', '')})
                for sess in json.loads(response.text):
                    if sess['kernel']['id'] == kernel_id:
                        return os.path.join(server['notebook_dir'], sess['notebook']['path'])

        def _subfn_vscode_jupyter_extract_notebook_path(module_locals=None) -> Optional[str]:
            """ extracts the path of the currently running jupyter notebook
            https://stackoverflow.com/a/75683730/9732163
            Usage:
                MUST be called like:
                notebook_name = IPythonHelpers._vscode_jupyter_extract_notebook_path(IPython.extract_module_locals())

            Returns:
                'halechr/repos/Spike3D/SCRATCH/2023-11-13 - Programmatic ipynb Processing.ipynb'
            """
            assert module_locals is not None, f"Must be called like: `notebook_name = _vscode_jupyter_extract_notebook_path(IPython.extract_module_locals())`"
            try:
                return "/".join(module_locals[1]["__vsc_ipynb_file__"].split("/")[-5:])
            except KeyError:
                return None

        ## BEGIN FUNCTION BODY:
        if module_locals is not None:
            notebook_path = _subfn_vscode_jupyter_extract_notebook_path(module_locals=module_locals)
            if notebook_path is not None:
                return notebook_path
        else:
            print(f'WARNING: module_locals is None so VSCode method will beskipped! Call like: notebook_name = IPythonHelpers.try_find_notebook_filepath(IPython.extract_module_locals())')
        # VSCode version didn't work.
        notebook_path = _subfn_try_find_notebook_fp_using_javascript()
        if notebook_path is not None:
            return notebook_path

        notebook_path = _subfn_try_find_notebook_fp_from_server()
        if notebook_path is not None:
            return notebook_path

        if notebook_path is None:
            print(f'WARNING: no method worked!')

        return notebook_path


    @classmethod
    def cell_vars(cls, get_globals_fn=None, offset=0):
        """ Captures a dictionary containing all assigned variables from the notebook cell it's used in.

        NOTE: You MUST call it with `captured_cell_vars = IPythonHelpers.cell_vars(lambda: globals())` if you want to access it in a notebook cell.

        Arguments:
            get_globals_fn: Callable - required for use in a Jupyter Notebook to access the correct globals (see ISSUE 2023-05-10


        Source:
            https://stackoverflow.com/questions/46824287/print-all-variables-defined-in-one-jupyter-cell
            BenedictWilkins
            answered Jun 12, 2020 at 15:55

        Usage:
            from pyphocorehelpers.programming_helpers import IPythonHelpers

            a = 1
            b = 2
            c = 3

            captured_cell_vars = IPythonHelpers.cell_vars(lambda: globals())

            >> {'a': 1, 'b': 2, 'c': 3}


        SOLVED ISSUE 2023-05-10 - Doesn't currently work when imported into the notebook because globals() accesses the code's calling context (this module) and not the notebook.
            See discussion here: https://stackoverflow.com/questions/61125218/python-calling-a-module-function-which-uses-globals-but-it-only-pulls-global
            https://stackoverflow.com/a/1095621/5646962

            Tangentially relevent:
                https://stackoverflow.com/questions/37718907/variable-explorer-in-jupyter-notebook

            Alternative approaches:
                https://stackoverflow.com/questions/27952428/programmatically-get-current-ipython-notebook-cell-output/27952661#27952661


        """
        from ipywidgets import get_ipython # required for IPythonHelpers.cell_vars
        import io
        from contextlib import redirect_stdout

        ipy = get_ipython()
        out = io.StringIO()

        with redirect_stdout(out):
            # ipy.magic("history {0}".format(ipy.execution_count - offset)) # depricated since IPython 0.13
            ipy.run_line_magic("history", format(ipy.execution_count - offset))

        #process each line...
        x = out.getvalue().replace(" ", "").split("\n")
        x = [a.split("=")[0] for a in x if "=" in a] #all of the variables in the cell

        if get_globals_fn is None:
            g = cls._helper_get_global_dict()
        else:
            g = get_globals_fn()

        result = {k:g[k] for k in x if k in g}
        return result

    @classmethod
    def extract_cells(cls, notebook_path: Union[Path,str]):
        """ extracts the cells from the provided notebook.
        # Example usage
        notebook_path = '../BatchGenerateOutputs_2023-11-13.ipynb'
        extracted_cells = IPythonHelpers.extract_cells(notebook_path)
        print(extracted_cells)

        cells_with_tags = []
        for cell in extracted_cells:
            cell_content = cell['source']
            cell_tags = cell['metadata'].get('tags', None)
            if (cell_tags is not None) and (len(cell_tags) > 0):
                cells_with_tags.append({'content': cell_content, 'tags': cell_tags})

        cells_with_tags
        """
        with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
            notebook_content = nbformat.read(notebook_file, as_version=4)

        cells = notebook_content['cells']
        return cells


    @classmethod
    def write_notebook(cls, cells, path: Path):
        """ writes a new notebook with the provided cells to the path provided. """
        if not isinstance(path, Path):
            path = Path(path).resolve()

        nb = nbformat.v4.new_notebook()

        # Assign the cells to the new notebook
        nb['cells'] = cells

        # Write the notebook to the given path
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        print(f"Notebook written to: {path}")



@define(slots=False)
class NotebookProcessor:
    """ processes Jupyter Notebooks
    from pyphocorehelpers.programming_helpers import NotebookProcessor

    notebook_path = Path(r"C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2024-01-22.ipynb").resolve()
    processor = NotebookProcessor(path=notebook_path)

    """
    path: Path = field()
    cells: List = field(default=Factory(list))

    def __attrs_post_init__(self):
        self.load_cells()

    def load_cells(self):
        self.cells = IPythonHelpers.extract_cells(self.path)
        print(self.cells)

    def get_cells_with_tags(self):
        return [{'content': cell['source'], 'tags': cell['metadata'].get('tags', [])}
                 for cell in self.cells if cell['metadata'].get('tags', [])]

    def get_cells_with_tag(self, tag):
        return [{'content': cell['source'], 'tags': cell['metadata'].get('tags')}
                for cell in self.cells if (tag in cell['metadata'].get('tags', []))]

    def get_empty_cells(self):
        return [cell for cell in self.cells if not cell['source'] or cell['source'].isspace()]


    def remove_empty_cells_and_save(self, new_path):
        """
        ## Remove all empty cells, and save the resultant notebook as the current notebook with the '_cleaned' filename suffix (but same extention)
        new_path = processor.path.with_stem(f'{processor.path.stem}_cleaned').resolve()
        processor.remove_empty_cells_and_save(new_path=new_path)


        """
        # spawn a new list omitting empty cells
        original_n_cells = len(self.cells)
        cleaned_cells = [cell for cell in self.cells if cell['source'] and not cell['source'].isspace()]
        post_clean_n_cells = len(cleaned_cells)
        n_changed_cells = original_n_cells - post_clean_n_cells
        if n_changed_cells > 0:
            print(f'original_n_cells: {original_n_cells}, post_clean_n_cells: {post_clean_n_cells}, n_changed_cells: {n_changed_cells} cells changed. Saving to {new_path}...')
            # Commit changes back to a notebook
            IPythonHelpers.write_notebook(cleaned_cells, new_path)
            print(f"Cleaned notebook saved to: {new_path}")
        else:
            print(f'no cells changed.')


    # from IPython.display import display, Javascript

    # def add_cell_below():
    # 	# js_code = """
    # 	# var cell = Jupyter.notebook.insert_cell_below();
    # 	# """
    # 	# display(Javascript(js_code))
    # 	## VSCode:
    # 	display({
    # 	"cell.insertCodeBelow": {
    # 		"code": 'print("This is a new cell")'
    # 	}
    # 	}, raw=True)
                


# 	def transform_dict_literal_to_constructor(dict_literal):
# 		# Regex pattern to match key-value pairs in dictionary literal syntax
# 		pattern = r"'(\w+)':([^,}]+)"

# 		# Find all matches of key-value pairs
# 		matches = re.findall(pattern, dict_literal)

# 		# Construct the transformed dictionary using dict() constructor syntax
# 		transformed_dict = "dict("
# 		for match in matches:
# 			key = match[0]
# 			value = match[1]
# 			transformed_dict += f"{key}={value},"

# 		transformed_dict = transformed_dict.rstrip(",")  # Remove trailing comma
# 		transformed_dict += ")"


class PythonDictionaryDefinitionFormat(Enum):
    """Enumeration for Python dictionary definition formats.
    TODO 2023-05-16: UNTESTED, UNUSED
    Goal: Transform code between Python's dictionary literal format:

        dictionary literal format:  {'require_intersecting_epoch':session.ripple, 'min_epoch_included_duration': 0.06, 'max_epoch_included_duration': None, 'maximum_speed_thresh': None, 'min_inclusion_fr_active_thresh': 0.01, 'min_num_unique_aclu_inclusions': 3}

        dict constructor format:    dict(require_intersecting_epoch=session.ripple, min_epoch_included_duration=0.06, max_epoch_included_duration=None, maximum_speed_thresh=None, min_inclusion_fr_active_thresh=0.01, min_num_unique_aclu_inclusions=3)


        from pyphocorehelpers.programming_helpers import PythonDictionaryDefinitionFormat
        input_str = "{'require_intersecting_epoch':session.ripple, 'min_epoch_included_duration': 0.06, 'max_epoch_included_duration': None, 'maximum_speed_thresh': None, 'min_inclusion_fr_active_thresh': 0.01, 'min_num_unique_aclu_inclusions': 3}"
        format_detected = PythonDictionaryDefinitionFormat.DICTIONARY_LITERAL

        converted_str = PythonDictionaryDefinitionFormat.convert_format(input_str, PythonDictionaryDefinitionFormat.DICT_CONSTRUCTOR)
        print(converted_str)
        # Output: dict(require_intersecting_epoch=session.ripple, min_epoch_included_duration=0.06, max_epoch_included_duration=None, maximum_speed_thresh=None, min_inclusion_fr_active_thresh=0.01, min_num_unique_aclu_inclusions=3)

    """

    DICTIONARY_LITERAL = "dictionary_literal"
    DICT_CONSTRUCTOR = "dict_constructor"

    @staticmethod
    def convert_format(input_str, target_format):
        """Converts the input string to the target format."""

        if target_format == PythonDictionaryDefinitionFormat.DICTIONARY_LITERAL:
            # Convert from dict() constructor to dictionary literal
            pattern = r"(\w+)=(\S+)"
            transformed_str = "{"
            for match in re.finditer(pattern, input_str):
                key = match.group(1)
                value = match.group(2)
                transformed_str += f"'{key}':{value}, "
            transformed_str = transformed_str.rstrip(", ")
            transformed_str += "}"
            return transformed_str

        elif target_format == PythonDictionaryDefinitionFormat.DICT_CONSTRUCTOR:
            # Convert from dictionary literal to dict() constructor
            pattern = r"'(\w+)':([^,}]+)"
            transformed_str = "dict("
            for match in re.finditer(pattern, input_str):
                key = match.group(1)
                value = match.group(2)
                transformed_str += f"{key}={value}, "
            transformed_str = transformed_str.rstrip(", ")
            transformed_str += ")"
            return transformed_str

        else:
            raise ValueError("Invalid target format specified.")






@contextlib.contextmanager
def disable_function_context(obj, fn_name: str):
    """ Disables a function within a context manager

    https://stackoverflow.com/questions/10388411/possible-to-globally-replace-a-function-with-a-context-manager-in-python

    Could be used for plt.show().
    ```python

    from pyphocorehelpers.programming_helpers import override_function_context

    with disable_function_context(plt, "show"):
        run_me(x)

    """
    temp = getattr(obj, fn_name)
    setattr(obj, fn_name, lambda: None)
    yield
    setattr(obj, fn_name, temp)




@contextlib.contextmanager
def override_function_context(obj, fn_name: str, override_defn):
    """ Overrides a function's definition with a different one within a context manager

    https://stackoverflow.com/questions/10388411/possible-to-globally-replace-a-function-with-a-context-manager-in-python

    Could be used for plt.show().
    ```python

    from pyphocorehelpers.programming_helpers import override_function_context

    with override_function_context(plt, "show", custom_print):
        run_me(x)

    """
    temp = getattr(obj, fn_name)
    setattr(obj, fn_name, override_defn)
    yield
    setattr(obj, fn_name, temp)


class MemoryManagement:
    """
    from pyphocorehelpers.programming_helpers import MemoryManagement

    args = MemoryManagement.deduplicate_memory_references(args)


    """

    @classmethod
    def print_memory_references(cls, *args) -> bool:
        """Check for duplicated memory references in the configs first

        from pyphocorehelpers.programming_helpers import MemoryManagement


        MemoryManagement.print_memory_references()


        """
        return [id(a_config) for a_config in args] # YUP, they're different for odd/even but duplicated for long/short

    @classmethod
    def has_duplicated_memory_references(cls, *args) -> bool:
        """Check for duplicated memory references in the configs first

        from pyphocorehelpers.programming_helpers import MemoryManagement


        MemoryManagement.has_duplicated_memory_references()


        """
        memory_ids = [id(a_config) for a_config in args] # YUP, they're different for odd/even but duplicated for long/short
        has_duplicated_reference: bool = len(np.unique(memory_ids)) < len(memory_ids)
        return has_duplicated_reference

    @classmethod
    def deduplicate_memory_references(cls, *args) -> list:
        """ Ensures that all entries in the args list point to unique memory addresses, deduplicating them with `deepcopy` if needed.

        Usage:

            from pyphocorehelpers.programming_helpers import MemoryManagement

            args = MemoryManagement.deduplicate_memory_references(args)

        """
        has_duplicated_reference: bool = cls.has_duplicated_memory_references(*args)
        if has_duplicated_reference:
            de_deuped_args = [deepcopy(v) for v in args]
            assert not cls.has_duplicated_memory_references(*de_deuped_args), f"duplicate memory references still exist even after de-duplicating with deepcopy!!!"
            return de_deuped_args
        else:
            return args
        

FunctionInspectionTuple = namedtuple('FunctionInspectionTuple', ['full_fn_spec', 'positional_args_names', 'kwargs_names', 'default_kwargs_dict'])

def inspect_callable_arguments(a_callable: Callable, debug_print=False) -> FunctionInspectionTuple:
    """ Inspects a callable's arguments

    Usage:        
        from pyphocorehelpers.programming_helpers import inspect_callable_arguments, FunctionInspectionTuple

        fn_inspect_tuple: FunctionInspectionTuple = inspect_callable_arguments(PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_track_templates)
        fn_inspect_tuple

        # FunctionInspectionTuple(full_fn_spec=FullArgSpec(args=['cls', 'curr_active_pipeline', 'track_templates', 'decoder_decoded_epochs_result_dict', 'epochs_name', 'included_epoch_indicies', 'name', 'title', 'defer_show'], varargs=None, varkw='kwargs',
        #                                                   defaults=('laps', None, 'CombinedDirectionalDecoderDecodedEpochsWindow', 'Pho Combined Directional Decoder Decoded Epochs', False), kwonlyargs=[], kwonlydefaults=None, annotations={'epochs_name': <class 'str'>}),
        #                         positional_args_names=['cls', 'curr_active_pipeline', 'track_templates', 'decoder_decoded_epochs_result_dict'],
        #                         kwargs_names=['epochs_name', 'included_epoch_indicies', 'name', 'title', 'defer_show'],
        #                         default_kwargs_dict={'epochs_name': 'laps', 'included_epoch_indicies': None, 'name': 'CombinedDirectionalDecoderDecodedEpochsWindow', 'title': 'Pho Combined Directional Decoder Decoded Epochs', 'defer_show': False})
                


    Progress:
        import inspect
        from neuropy.plotting.ratemaps import plot_ratemap_1D, plot_ratemap_2D
        from pyphocorehelpers.programming_helpers import inspect_callable_arguments
        
        fn_spec = inspect.getfullargspec(plot_ratemap_2D)
        fn_sig = inspect.signature(plot_ratemap_2D)
        ?fn_sig

            # fn_sig
        dict(fn_sig.parameters)
        # fn_sig.parameters.values()

        fn_sig.parameters['plot_mode']
        # fn_sig.parameters
        fn_spec.args # all kwarg arguments: ['x', 'y', 'num_bins', 'debug_print']

        fn_spec.defaults[-2].__class__.__name__ # a tuple of default values corresponding to each argument in args; ((64, 64), False)
    """
    import inspect
    full_fn_spec = inspect.getfullargspec(a_callable) # FullArgSpec(args=['item1', 'item2', 'item3'], varargs=None, varkw=None, defaults=(None, '', 5.0), kwonlyargs=[], kwonlydefaults=None, annotations={})
    # fn_sig = inspect.signature(compute_position_grid_bin_size)
    if debug_print:
        print(f'fn_spec: {full_fn_spec}')
    # fn_spec.args # ['item1', 'item2', 'item3']
    # fn_spec.defaults # (None, '', 5.0)

    num_positional_args = len(full_fn_spec.args) - len(full_fn_spec.defaults) # all kwargs have a default value, so if there are less defaults than args, than the first args must be positional args.
    positional_args_names = full_fn_spec.args[:num_positional_args] # [fn_spec.args[i] for i in np.arange(num_positional_args, )] np.arange(num_positional_args)
    kwargs_names = full_fn_spec.args[num_positional_args:] # [fn_spec.args[i] for i in np.arange(num_positional_args, )]
    if debug_print:
        print(f'fn_spec_positional_args_list: {positional_args_names}\nfn_spec_kwargs_list: {kwargs_names}')
    default_kwargs_dict = {argname:v for argname, v in zip(kwargs_names, full_fn_spec.defaults)} # {'item1': None, 'item2': '', 'item3': 5.0}

    return FunctionInspectionTuple(full_fn_spec=full_fn_spec, positional_args_names=positional_args_names, kwargs_names=kwargs_names, default_kwargs_dict=default_kwargs_dict)

def get_arguments_as_optional_dict(**kwargs):
    """ Easily converts your existing argument-list style default values into a dict:
            Defines a simple function that takes only **kwargs as its inputs and prints the values it recieves. Paste your values as arguments to the function call. The dictionary will be output to the console, so you can easily copy and paste.
        Usage:
            >>> get_arguments_as_optional_dict(point_size=8, font_size=10, name='build_center_labels_test', shape_opacity=0.8, show_points=False)

            Output: ", **({'point_size': 8, 'font_size': 10, 'name': 'build_center_labels_test', 'shape_opacity': 0.8, 'show_points': False} | kwargs)"
    """
    CodeConversion.get_arguments_as_optional_dict(**kwargs)
    
@unique
class GeneratedClassDefinitionType(ExtendedEnum):
    """Specifies which type of class to generate in CodeConversion.convert_dictionary_to_class_defn(...)."""
    STANDARD_CLASS = "STANDARD_CLASS"
    DATACLASS = "DATACLASS"
    ATTRS_CLASS = "ATTRS_CLASS"

    @property
    def class_decorators(self):
        return self.decoratorsList()[self]

    @property
    def class_required_imports(self):
        return self.requiredImportsList()[self]

    @property
    def include_init_fcn(self):
        return self.include_init_fcnList()[self]

    @property
    def include_properties_defns(self):
        return self.include_properties_defnsList()[self]

    # Static properties
    @classmethod
    def decoratorsList(cls):
        return cls.build_member_value_dict([None,"@dataclass","@define(slots=False)"])

    @classmethod
    def requiredImportsList(cls):
        return cls.build_member_value_dict([None,"from dataclasses import dataclass","from attrs import define, field, Factory, astuple, asdict"])

    @classmethod
    def include_init_fcnList(cls):
        return cls.build_member_value_dict([True, False, False])

    @classmethod
    def include_properties_defnsList(cls):
        return cls.build_member_value_dict([False, True, True])
    

import inspect
import re
import ast # SourceCodeParsing


@metadata_attributes(short_name=None, tags=['source-code-parsing', 'source-code'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 08:37', related_items=[])
class SourceCodeParsing:
    """ Contains functions that help parse python source code.
    
    Can be used in the future to enable VSCode coding automations like converting selected text between two formats, etc.


    
    """
    @classmethod
    @function_attributes(short_name=None, tags=['return', 'source-code-parsing', 'pho'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 08:38', related_items=[])
    def get_return_line_numbers(cls, func):
        """ Get the line numbers in the source code of a function where a 'return' statement appears. 
        
        from pyphocorehelpers.programming_helpers import SourceCodeParsing

        return_lines_info = SourceCodeParsing.get_last_return_lines(compute_pho_heuristic_replay_scores)
        for line_no, code in return_lines_info:
            print(f"Line {line_no}: {code}")
            
        """
        if not inspect.isfunction(func):
            raise ValueError('The provided object is not a function')

        source_lines, starting_line_no = inspect.getsourcelines(func)
        
        # Combine the source lines into a single string for pattern matching
        source = ''.join(source_lines)

        # Find all occurrences of return using a regular expression
        # This simplistic regex assumes that 'return' will be at the start of a line or after a space,
        # and that it will be followed by a space, a newline, a comment, or the end of a statement.
        # In reality, you may want to adjust this to handle more complex scenarios (like nested functions).
        return_lines = [
            match.start()
            for match in re.finditer(r'(?<![^\s])return(?![^\s])', source)
        ]

        line_numbers = [source[:line_start].count('\n') + starting_line_no for line_start in return_lines]

        return line_numbers

    @classmethod
    @function_attributes(short_name=None, tags=['ALT','source-code-parsing', 'pho', 'efficiency'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-07 08:36', related_items=[])
    def get_last_return_lines(cls, func):
        """ Get the line and code of the last return statement of each block in a function. """
        if not inspect.isfunction(func):
            raise ValueError('The provided object is not a function')

        source_lines, starting_line_no = inspect.getsourcelines(func)
        # Parse the source code into an AST
        func_ast = ast.parse(''.join(source_lines), mode='exec')

        return_statements = []

        # Helper function to visit Return nodes in the AST
        class ReturnVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # In each function, we want only the last 'Return' node in each block, if it exists
                for body_element in node.body:
                    if isinstance(body_element, (ast.For, ast.While, ast.If)):
                        self.generic_visit(body_element)
                    elif isinstance(body_element, ast.Return):
                        return_statements.append((body_element, starting_line_no + body_element.lineno - 1))

            def visit_If(self, node):
                self.visit_branch(node)

            def visit_For(self, node):
                self.visit_branch(node)

            def visit_While(self, node):
                self.visit_branch(node)

            def visit_branch(self, branch_node):
                for body in [branch_node.body, branch_node.orelse]:
                    if body: # it's not an empty body or orelse
                        last_stmt = body[-1]
                        if isinstance(last_stmt, ast.Return):
                            return_statements.append((last_stmt, starting_line_no + last_stmt.lineno - 1))
                        else:
                            self.generic_visit(last_stmt)

        ReturnVisitor().visit(func_ast)

        return_lines_info = [(lineno, source_lines[lineno - starting_line_no].strip()) 
                            for _, lineno in return_statements]

        return return_lines_info


class CodeConversion(object):
    """ Converts code (usually passed as text) to various alternative formats to ease development workflows.
    from pyphocorehelpers.programming_helpers import CodeConversion

    TODO 2023-10-24 - Add Ignored imports:
    ignored_imports = ['import bool,
        "import str",
        "import tuple",
        "import list",
        "import dict",
        ]

    substitution_dict = {'pathlib.WindowsPath':'pathlib.Path',
    'pathlib.PosixPath':'pathlib.Path'
    }


    # Definition Lines: __________________________________________________________________________________________________ #
    ## a multiline string containing lines of valid python code definitions
    test_parameters_defns_code_string = '''
                max_num_spikes_per_neuron = 20000 # the number of spikes to truncate each neuron's timeseries to
                kleinberg_parameters = DynamicParameters(s=2, gamma=0.1)
                use_progress_bar = False # whether to use a tqdm progress bar
                debug_print = False # whether to print debug-level progress using traditional print(...) statements
            '''
    ## Functions: `convert_defn_lines_to_dictionary(...)`, `convert_defn_lines_to_parameters_list(...)`, `convert_defn_lines_to_parameters_list(...)`,

    # Dictionary: ________________________________________________________________________________________________________ #
        {'spike_raster_plt_2d': <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x168558703a0>,
                'spike_raster_plt_3d': <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster.Spike3DRaster at 0x1673e722310>,
                'spike_raster_window': <pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget.Spike3DRasterWindowWidget at 0x1673e7aaa60>}



    """
    # _types_replace_dict = {'numpy.':'np.', 'pandas.':'pd.'}
    
    _types_replace_dict = {'numpy.':'np.', 'pandas.':'pd.'} # OLD
    # _types_replace_dict = {{'dict': 'Dict', 'list': 'List', 'tuple': 'Tuple', 'numpy.ndarray': 'NDArray', 'numpy.':'np.', 'pandas.':'pd.'}}
    # _types_import_replace_dict = {'import dict': 'from typing import Dict', 'import list': 'from typing import List', 'import tuple': 'from typing import Tuple', 'import numpy.ndarray': 'from nptyping import NDArray'}
    _inverse_types_replace_dict = {v:k for k,v in _types_replace_dict.items()}

    # types_replace_dict.update({'dict':'Dict'})
    # types_import_replace_dict.update({'import dict':'from typing import Dict'})

    # types_replace_dict.update({'list':'List'})
    # types_import_replace_dict.update({'import list':'from typing import List'})

    # types_replace_dict.update({'tuple':'Tuple'})
    # types_import_replace_dict.update({'import tuple':'from typing import Tuple'})

    # types_replace_dict.update({'np.ndarray':'NDArray'})
    # types_import_replace_dict.update({'import np.ndarray':'from nptyping import NDArray'})
                
            

    _general_find_replace_dict = {'pd.core.frame.DataFrame':'pd.DataFrame'}


    @classmethod
    def apply_find_replace(cls, find_replace_dict: dict, target_str: str):
        """ returns the target_str after applying each find/replace pair in find_replace_dict to it.

        Usage:

            cls.apply_find_replace(find_replace_dict=cls._general_find_replace_dict, target_str=)
        """
        if find_replace_dict is None:
            find_replace_dict = cls._general_find_replace_dict # get default
        for find_txt, replace_txt in find_replace_dict.items():
            target_str = target_str.replace(find_txt, replace_txt)
        return target_str


    @classmethod
    def _stripComments(cls, code):
        code = str(code)
        # any '#' comment (not just new lines
        any_line_comment_format = r'(?m) *#.*\n?'
        standalone_full_line_comment_only_format = r'(?m)^ *#.*\n?' # matches only lines that start with a line comment
        end_line_only_comment_format = r'(?m)(?!\s*\w+\s)+#.*\n?' # Match only the comments at the end of lines, actually also includes the comment-only lines

        # format_str = standalone_full_line_comment_only_format
        format_str = end_line_only_comment_format
        return re.sub(format_str, '', code)

    @classmethod
    def _match_and_parse_defn_line(cls, code_line):
        """Converts a single line of code that defines a variable to a dictionary consisting of {'var_name': str, 'equals_portion': str, 'end_portion': str}

        Args:
            code_line (_type_): _description_

        Returns:
            _type_: _description_
        """
        code_line = str(code_line)
        format_str = r'^\s*(?P<var_name>\w+)(?P<equals_portion>\s*=\s*)(?P<end_portion>.+)\n?' # gets the start of the line followed by any amount of whitespace followed by a variable name followed by an equals sign
        m = re.match(format_str, code_line)
        if m is None:
            return None
        else:
            matches_group = m.groupdict()  # {'var_name': 'Malcolm', 'equals_portion': 'Reynolds', 'end_portion': ''}
            return matches_group

    @classmethod
    def _convert_defn_line_to_dictionary_line(cls, code_line):
        """Converts a single line of code that defines a variable to a line of a dictionary.
        """
        code_line = str(code_line)
        matches_dict = cls._match_and_parse_defn_line(code_line)
        if matches_dict is None:
            return ''
        else:
            # Format as dictionary:
            dict_line = f"'{matches_dict['var_name'].strip()}': {matches_dict['end_portion'].strip()}"
            return dict_line

    @classmethod
    def _convert_defn_line_to_parameter_list_item(cls, code_line):
        """Converts a single line of code that defines a variable to a single entry in a parameter list.
        """
        code_line = str(code_line)
        matches_dict = cls._match_and_parse_defn_line(code_line)
        if matches_dict is None:
            return ''
        else:
            # Format as entry in a parameter list:
            param_item = f"{matches_dict['var_name'].strip()}={matches_dict['end_portion'].strip()}"
            return param_item

    @classmethod
    def _isinstance_namedtuple(cls, an_obj_instance) -> bool:
        """ Checks if an object instance is a subclass of `namedtuple`.
        `isinstance(an_obj_instance, (namedtuple, ))` does not work because namedtuple is a generic type, so we'll check for being a subclass of `tuple` and the presence of the `_asdict` method
        Replacement for: `isinstance(an_obj_instance, (namedtuple, ))`

        Usage:
        if cls._isinstance_namedtuple(an_obj_instance):
            # use namedtuple's built-in `._asdict()` property:
            an_obj_instance_dict = an_obj_instance._asdict()
        else:
            print(f'not a namedtuple subclass')
        """
        return (isinstance(an_obj_instance, (tuple, )) and hasattr(an_obj_instance, '_asdict'))

    @classmethod
    def _try_parse_to_dictionary_if_needed(cls, target_dict) -> dict:
        """ returns a for-sure dictionary or throws an Exception

        target_dict: either a dictionary object or a string of code that defines a dictionary object (such as "{'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield}")

        """
        if isinstance(target_dict, str):
            # if the target_dict is a string instead of a dictionary, assume it is code that defines a dictionary
            try:
                target_dict = cls.build_dummy_dictionary_from_defn_code(target_dict)
            except Exception as e:
                print(f'ERROR: Could not convert code string: {target_dict} to a proper dictionary! Exception: {e}')
                raise e
        elif cls._isinstance_namedtuple(target_dict):
            # use namedtuple's built-in `._asdict()` property:
            target_dict = target_dict._asdict()

        assert isinstance(target_dict, dict), f"target_dict must be a dictionary but is of type: {type(target_dict)},\ntarget_dict: {target_dict}"
        return target_dict # returns a for-sure dictionary or throws an Exception

    @classmethod
    def _find_best_type_representation_string(cls, a_type, unspecified_generic_type_name='type', keep_generic_types=['NoneType'], types_replace_dict = {'numpy.':'np.', 'pandas.':'pd.'}) -> str:
        """ Uses `strip_type_str_to_classname(a_type) to find the best type-string representation.

        Usage:
            from pyphocorehelpers.programming_helpers import CodeConversion

            CodeConversion._find_best_type_representation_string(str(type(k)))


        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname # used to convert dict to class with types
        out_extracted_typestring = strip_type_str_to_classname(a_type_str=a_type)
        if out_extracted_typestring in keep_generic_types:
            # If the parsed typestring is in the ignored type-string list, keep the generic type_name instead of this one.
            out_extracted_typestring = unspecified_generic_type_name

        for find_str, rep_str in types_replace_dict.items():
            out_extracted_typestring = out_extracted_typestring.replace(find_str, rep_str) # replace find_str with rep_str
            # e.g. out_extracted_typestring.replace('numpy.', 'np.') # replace 'numpy.' with 'np.'

        return out_extracted_typestring

    @classmethod
    def split_type_str(cls, type_str):
        """ for type_str = 'neuropy.core.epoch.Epoch' """
        base_type_str = '.'.join(type_str.split('.')[:-2]) # 'neuropy.core.epoch'
        class_name = type_str.split('.')[-1]  # 'Epoch'
        return base_type_str, class_name

    @classmethod
    def get_import_statement_from_type_str(cls, type_str):
        """ for type_str = 'neuropy.core.epoch.Epoch' """
        split_type_components = type_str.split('.')
        num_items = len(split_type_components)
        base_type_str = '.'.join(split_type_components[:-2]) # 'neuropy.core.epoch'
        class_name = split_type_components[-1]  # 'Epoch'
        import_statement = f'from {base_type_str} import {class_name}' # 'from neuropy.core.epoch import Epoch'
        return import_statement

    @classmethod
    def convert_type_to_typehint_string(cls, target_class_str: str, use_relative_types:bool=True) -> Tuple[str, Optional[str]]:
        """ returns the proper typestring for the provided target_class_str for use as typehints or elsewhere
        Usage:

        from pyphocorehelpers.programming_helpers import CodeConversion

        
        History: factored out of `CodeConversion.convert_dictionary_to_class_defn` on 2024-04-05.

        """
        if isinstance(target_class_str, type):
            target_class_str = str(target_class_str) # convert the type to a string

        # by default type(v) gives <class 'numpy.ndarray'>
        full_type_string: str = f"{cls._find_best_type_representation_string(target_class_str)}"
        output_type_string: str = None
        import_statement: Optional[str] = None

        if use_relative_types:
            """ for type_str = 'neuropy.core.epoch.Epoch' """
            split_type_components = full_type_string.split('.')
            num_items = len(split_type_components)
            if num_items == 1:
                base_type_str = None
                class_name = split_type_components[0]
                import_statement = f'import {class_name}' # 'from neuropy.core.epoch import Epoch'
            elif num_items > 1:
                base_type_str = '.'.join(split_type_components[:-1]) # 'neuropy.core.epoch'
                class_name = split_type_components[-1]  # 'Epoch'
                import_statement = f'from {base_type_str} import {class_name}' # 'from neuropy.core.epoch import Epoch'
            else:
                raise NotImplementedError

            if split_type_components[0] in ['np', 'pd']:
                # do different for pandas and numpy
                relative_type_str: str = full_type_string
                import_statement = None # f'import {split_type_components[0]}' # 'import numpy as np' TODO: import numpy/pd
            else:
                relative_type_str: str = class_name

            # Apply the find/replace dict to fix issues like '' being output
            relative_type_str: str = cls.apply_find_replace(find_replace_dict=cls._general_find_replace_dict, target_str=relative_type_str)
            output_type_string = relative_type_str
        else:
            output_type_string = full_type_string # just use the full type string

        return output_type_string, import_statement 
            

    @classmethod
    def get_dict_typehint_string(cls, a_dict: Dict, use_relative_types:bool = True) -> str:
        """ Generates the typehint from a dictionary, including its 1-layer nested datatypes (returns 'Dict[str, pd.DataFrame]' instead of 'Dict', for example.
         
        :return - a typehint string like 'Dict[str, pd.DataFrame]'
        
        """
        from neuropy.utils.indexing_helpers import collapse_if_identical

        assert isinstance(a_dict, dict)
        # note `[0]` in the following just gets the typestring itself, and not the import that is produced.
        _collapsed_output = collapse_if_identical([(cls.convert_type_to_typehint_string(type(k), use_relative_types=use_relative_types)[0], cls.convert_type_to_typehint_string(type(v), use_relative_types=use_relative_types)[0]) for k,v in a_dict.items()])
        return f"Dict[{', '.join(_collapsed_output)}]" # 'Dict[str, pd.DataFrame]'
    


    @classmethod
    def get_tuple_typehint_string(cls, a_tuple: Tuple, use_relative_types:bool = True) -> str:
        """ Generates the typehint from a tuple, including its 1-layer nested datatypes (returns 'Dict[str, pd.DataFrame]' instead of 'Dict', for example.
         
        :return - a typehint string like 'Dict[str, pd.DataFrame]'
        
        """
        assert isinstance(a_tuple, tuple)
        # note `[0]` in the following just gets the typestring itself, and not the import that is produced.
        _collapsed_output = [cls.convert_type_to_typehint_string(type(v), use_relative_types=use_relative_types)[0] for v in a_tuple] # DO NOT collapse
        return f"Tuple[{', '.join(_collapsed_output)}]" # 'Dict[str, pd.DataFrame]'




    # ==================================================================================================================== #
    # Public/Main Methods                                                                                                  #
    # ==================================================================================================================== #

    @classmethod
    def convert_dictionary_to_defn_lines(cls, target_dict, multiline_assignment_code=False, dictionary_name:str='target_dict', include_comment:bool=True, copy_to_clipboard=True, output_variable_prefix=''):
        """ The reciprocal operation of convert_defn_lines_to_dictionary
            target_dict: either a dictionary object or a string of code that defines a dictionary object (such as "{'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield}")
            multiline_assignment_code: if True, generates a separate line for each assignment, otherwise assignment is done inline
            dictionary_name: the name to use for the dictionary in the generated code

        Examples:
            from pyphocorehelpers.general_helpers import CodeConversion
            curr_active_pipeline.last_added_display_output
            >>> {'spike_raster_plt_2d': <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x168558703a0>,
                'spike_raster_plt_3d': <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster.Spike3DRaster at 0x1673e722310>,
                'spike_raster_window': <pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget.Spike3DRasterWindowWidget at 0x1673e7aaa60>}

            convert_dictionary_to_defn_lines(curr_active_pipeline.last_added_display_output, multiline_assignment_code=False, dictionary_name='curr_active_pipeline.last_added_display_output')
            >>> spike_raster_plt_2d, spike_raster_plt_3d, spike_raster_window = curr_active_pipeline.last_added_display_output['spike_raster_plt_2d'], curr_active_pipeline.last_added_display_output['spike_raster_plt_3d'], curr_active_pipeline.last_added_display_output['spike_raster_window'] # Extract variables from the `curr_active_pipeline.last_added_display_output` dictionary to the local workspace

            active_str = convert_dictionary_to_defn_lines(curr_active_pipeline.last_added_display_output, multiline_assignment_code=True, dictionary_name='curr_active_pipeline.last_added_display_output')
            active_str
            >>> "# Extract variables from the `curr_active_pipeline.last_added_display_output` dictionary to the local workspace\nspike_raster_plt_2d = curr_active_pipeline.last_added_display_output['spike_raster_plt_2d']\nspike_raster_plt_3d = curr_active_pipeline.last_added_display_output['spike_raster_plt_3d']\nspike_raster_window = curr_active_pipeline.last_added_display_output['spike_raster_window']"

            print(active_str)
            >>>
                # Extract variables from the `curr_active_pipeline.last_added_display_output` dictionary to the local workspace
                spike_raster_plt_2d = curr_active_pipeline.last_added_display_output['spike_raster_plt_2d']
                spike_raster_plt_3d = curr_active_pipeline.last_added_display_output['spike_raster_plt_3d']
                spike_raster_window = curr_active_pipeline.last_added_display_output['spike_raster_window']


        Example 2:
            dictionary_definition_string = "{'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield}"
            CodeConversion.convert_dictionary_to_defn_lines(dictionary_definition_string, dictionary_name='curr_axs_dict')
            >>>
                firing_rate, lap_spikes, placefield = curr_axs_dict['firing_rate'], curr_axs_dict['lap_spikes'], curr_axs_dict['placefield'] # Extract variables from the `curr_axs_dict` dictionary to the local workspace
        """
        target_dict = cls._try_parse_to_dictionary_if_needed(target_dict=target_dict) # Ensure a valid dictionary is provided or can be built
        comment_str = f"# Extract variables from the `{dictionary_name}` dictionary to the local workspace"
        if multiline_assignment_code:
            # Separate line per assignment
            """
            # Extract variables from the `{dictionary_name}` dictionary to the local workspace
            spike_raster_plt_2d = target_dict['spike_raster_plt_2d']
            spike_raster_plt_3d = target_dict['spike_raster_plt_3d']
            spike_raster_window = target_dict['spike_raster_window']

            """
            code_str = '\n'.join([f"{output_variable_prefix}{k} = {dictionary_name}['{k}']" for k,v in target_dict.items()])

            if include_comment:
                code_str = f"{comment_str}\n{code_str}" # add comment above code

        else:
            # Generates an inline assignment, e.g. "spike_raster_plt_2d, spike_raster_plt_3d, spike_raster_window = target_dict['spike_raster_plt_2d'], target_dict['spike_raster_plt_3d'], target_dict['spike_raster_window']"
            """ Assignment all on a single line
            spike_raster_plt_2d, spike_raster_plt_3d, spike_raster_window = target_dict['spike_raster_plt_2d'], target_dict['spike_raster_plt_3d'], target_dict['spike_raster_window'] # Extract variables from the `target_dict` dictionary to the local workspace
            """
            vars_name_list = list(target_dict.keys())
            rhs_code_str = ', '.join([f"{dictionary_name}['{k}']" for k in vars_name_list])
            lhs_code_str = ', '.join([f"{output_variable_prefix}{k}" for k in vars_name_list])
            code_str = f'{lhs_code_str} = {rhs_code_str}'
            if include_comment:
                code_str = f"{code_str} {comment_str}" # add comment at the very end of the code line

        if copy_to_clipboard:
            df = pd.DataFrame([code_str])
            df.to_clipboard(index=False, header=False, sep=',')
            print(f'Copied "{code_str}" to clipboard!')

        return code_str

    @classmethod
    def convert_dictionary_to_class_defn(cls, target_dict, class_name:str='TargetClass', class_definition_mode:GeneratedClassDefinitionType = None, copy_to_clipboard=True, output_variable_prefix='', class_decorators="@dataclass",
        include_init_fcn=True, include_initializer_default_values=False, include_properties_defns=False, include_types=True, use_relative_types=True,
        indent_character='    ', pre_class_spacing='\n', post_class_spacing='\n\n'):
        """ Builds a class definition from a target_dict
            target_dict: either a dictionary object or a string of code that defines a dictionary object (such as "{'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield}")
            multiline_assignment_code: if True, generates a separate line for each assignment, otherwise assignment is done inline
            class_name: the name to use for the generated class in the generated code


            include_types: bool - If True, type hints are included in both the properties_defns and init_fcn (if those are enabled, otherwise changes nothing)

            include_properties_defns: if True, includes the properties at the top of the class definintion like is done in an @dataclass
            include_init_fcn: if True, includes a standard __init__(self, ...) function in the definintion




        Examples:
            from pyphocorehelpers.general_helpers import CodeConversion, GeneratedClassDefinitionType
            CodeConversion.convert_dictionary_to_class_defn(_out.to_dict(), class_name='PlacefieldSnapshot', indent_character='    ', include_types=True, class_decorators=None, include_initializer_default_values=False)

            >>>
            class PlacefieldSnapshot(object):
                # Docstring for PlacefieldSnapshot.

                def __init__(self, num_position_samples_occupancy: np.ndarray, seconds_occupancy: np.ndarray, spikes_maps_matrix: np.ndarray, smoothed_spikes_maps_matrix: type, occupancy_weighted_tuning_maps_matrix: np.ndarray):
                    super(PlacefieldSnapshot, self).__init__()        self.num_position_samples_occupancy = num_position_samples_occupancy
                    self.seconds_occupancy = seconds_occupancy
                    self.spikes_maps_matrix = spikes_maps_matrix
                    self.smoothed_spikes_maps_matrix = smoothed_spikes_maps_matrix
                    self.occupancy_weighted_tuning_maps_matrix = occupancy_weighted_tuning_maps_matrix



        """
        needed_import_statements = []
        target_dict = cls._try_parse_to_dictionary_if_needed(target_dict=target_dict) # Ensure a valid dictionary is provided or can be built

        if class_definition_mode is None:
            class_definition_mode =  GeneratedClassDefinitionType.ATTRS_CLASS # Default to an ATTRS_CLASS as of 2023-05-10

        if class_definition_mode is not None and isinstance(class_definition_mode, GeneratedClassDefinitionType):
            # generated class definition type provided to shortcut the settings
            print(f'WARNING: class_definition_mode ({class_definition_mode}) was provided, overriding the `class_decorators`,  `include_init_fcn`, `include_properties_defns` settings!')
            class_required_imports = class_definition_mode.class_required_imports
            needed_import_statements.append(class_required_imports)
            class_decorators = class_definition_mode.class_decorators
            include_init_fcn = class_definition_mode.include_init_fcn
            include_properties_defns = class_definition_mode.include_properties_defns

        if class_decorators is not None and len(class_decorators) > 0:
            class_decorators=f"{class_decorators}\n" # append the line break after the decorators
        else:
            class_decorators = '' # empty string

        comment_str = f'# Docstring for {class_name}. \n'
        # comment_str = f"\"\"\"Docstring for {dictionary_name}.\"\"\""
        # comment_str = f'"""Docstring for {dictionary_name}.""""""'
        # comment_str = f'\"\"\" Docstring for {class_name}. \"\"\"\n'

        class_header_code_str = f"{pre_class_spacing}{class_decorators}class {class_name}(object):\n{indent_character}{comment_str}"

        if include_properties_defns:
            """ if include_properties_defns is True, includes the properties at the top of the class definintion like is done in an @dataclass 

            """
            if include_types:
                # by default type(v) gives <class 'numpy.ndarray'>
                if use_relative_types:
                    # Using `cls.convert_type_to_typehint_string(...)`:
                    relative_types_dict = {k:cls.convert_type_to_typehint_string(type(v), use_relative_types=use_relative_types)[0] for k,v in target_dict.items()}
                    import_statements_list = [cls.convert_type_to_typehint_string(type(v), use_relative_types=use_relative_types)[1] for v in target_dict.values()]
                    for import_statement in import_statements_list:
                        if (import_statement is not None) and (import_statement not in needed_import_statements):
                            needed_import_statements.append(import_statement)
                    
                    member_properties_code_str = '\n'.join([f"{indent_character}{output_variable_prefix}{k}: {v}" for k,v in relative_types_dict.items()])
                else:
                    member_properties_code_str = '\n'.join([f"{indent_character}{output_variable_prefix}{k}: {cls._find_best_type_representation_string(type(v))}" for k,v in target_dict.items()])
            else:
                # leave generic 'type' placeholder for each
                member_properties_code_str = '\n'.join([f"{indent_character}{output_variable_prefix}{k}: type" for k,v in target_dict.items()])

            member_properties_code_str = f"\n{member_properties_code_str}"
        else:
            member_properties_code_str = ''


        if include_init_fcn:
            init_fcn_code_str = cls._build_class_init_fcn(target_dict, class_name=class_name, output_variable_prefix=output_variable_prefix, include_type_hinting=include_types, include_default_values=include_initializer_default_values, indent_character=indent_character)
            init_fcn_code_str = f"\n\n{init_fcn_code_str}"

        else:
            init_fcn_code_str = ''

        # Build the import statements:
        if len(needed_import_statements) > 0:
            import_statements_block = '\n'.join(needed_import_statements)
            class_header_code_str=f"{import_statements_block}\n{class_header_code_str}" # prepend the imports

        code_str: str = f"\n{class_header_code_str}{member_properties_code_str}{init_fcn_code_str}{post_class_spacing}" # add comment above code

        if copy_to_clipboard:
            df = pd.DataFrame([code_str])
            df.to_clipboard(index=False,header=False)
            print(f'Copied "{code_str}" to clipboard!')

        return code_str

    # ==================================================================================================================== #
    # Class/Static Methods                                                                                                 #
    # ==================================================================================================================== #

    @classmethod
    def _parse_NameError(cls, e):
        """Takes a NameError e and parses it into the name of the missing variable as a string

        Args:
            e (_type_): _description_

        Returns:
            _type_: _description_
        """
        # when e is a NameError, str(e) is a stirng like: "name 'curr_ax_firing_rate' is not defined"
        name_error_str = str(e)
        name_error_split_str = name_error_str.split("'")
        assert len(name_error_split_str)==3, f"name_error_split_str: {name_error_split_str}"
        missing_variable_name = name_error_split_str[1] # e.g. 'curr_ax_firing_rate'
        # print(f'\te.name: {e.name}')
        return missing_variable_name

    @classmethod
    def extract_undefined_variable_names_from_code(cls, code_dict_defn:str, max_iterations_before_abort:int=50, debug_print=False):
        """ Finds the names of all undefined variables in a given block of code by repetitively replacing it and re-evaluating it. Probably not the smartest doing this.
        Based on `cls.build_dummy_dictionary_from_defn_code(...)`


        Inputs:
            code_dict_defn: lines of code that define several python variables to be converted to dictionary entries
                e.g. code_dict_defn: "{'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield}"
        Outputs:
            a dictionary
        """
        did_complete = False
        num_iterations = 0
        last_exception = None
        output_undefined_variable_names = []
        while (num_iterations <= max_iterations_before_abort) and (not did_complete):
            try:
                # Tries to evaluate the code_dict_defn, which is just a string, into a valid dictionary object
                eval(code_dict_defn) # , None, None # should produce NameError: name 'curr_ax_firing_rate' is not defined
                did_complete = True
            except NameError as e:
                if debug_print:
                    print(f'iteration {num_iterations}: {e}')
                last_exception = e
                missing_variable_name = cls._parse_NameError(e)
                if debug_print:
                    print(f'missing_variable_name: {missing_variable_name}')
                output_undefined_variable_names.append(missing_variable_name)
                exec(f'{missing_variable_name} = None') # define the missing variable as None in this environment to continue without errors
            except Exception as e:
                # Unhandled/Unexpected exception:
                print(f'ERROR: iteration {num_iterations}: Unhandled exception: {e}')
                last_exception = e
                raise e
            num_iterations = num_iterations + 1

        if not did_complete:
            # still failed to execute, failed
            print(f'ERROR: Still failed to execute after {num_iterations}. Found output_undefined_variable_names: {output_undefined_variable_names}.')
            if last_exception is not None:
                raise last_exception
            else:
                raise NotImplementedError
        else:
            return output_undefined_variable_names


    @classmethod
    def build_dummy_dictionary_from_defn_code(cls, code_dict_defn:str, max_iterations_before_abort:int=50, missing_variable_values=None, debug_print=False):
        """ Consider an inline dictionary definition such as:

            # output the axes created:
            axs_list.append({'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield})

        The goal is to be able to extract the members of this dictionary from its returned value, effectively "unwrapping" the dictionary. The problem with just using convert_dictionary_to_defn_lines(...) directly is that you don't have a dictionary, you have a string the defines the dictionary in the original code.
        Attempting to build a dummy dictionary from this code doesn't work, as the variables aren't defined outside of the function that created them.

        One way around this problem is to do the following to define the unbound variables in the dictionary to None:

            curr_ax_firing_rate, curr_ax_lap_spikes, curr_ax_placefield = None, None, None
            CodeConversion.convert_dictionary_to_defn_lines({'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield}, dictionary_name='curr_axs_dict')

        This works, but requires extracting the variable names and assigning None for each one. This function automates that process with eval(...)

        >>>
            iteration 0: name 'curr_ax_firing_rate' is not defined
            missing_variable_name: curr_ax_firing_rate
            iteration 1: name 'curr_ax_lap_spikes' is not defined
            missing_variable_name: curr_ax_lap_spikes
            iteration 2: name 'curr_ax_placefield' is not defined
            missing_variable_name: curr_ax_placefield
            Copied "firing_rate, lap_spikes, placefield = target_dict['firing_rate'], target_dict['lap_spikes'], target_dict['placefield'] # Extract variables from the `target_dict` dictionary to the local workspace" to clipboard!

        Inputs:
            code_dict_defn: lines of code that define several python variables to be converted to dictionary entries
                e.g. code_dict_defn: "{'firing_rate':curr_ax_firing_rate, 'lap_spikes': curr_ax_lap_spikes, 'placefield': curr_ax_placefield}"
        Outputs:
            a dictionary
        """
        target_dict = None

        num_iterations = 0
        last_exception = None

        if missing_variable_values is None:
            # should be a string
            # missing_variable_values = {'*':'None'}
            missing_variable_fill_function = lambda var_name: 'None'
        elif isinstance(missing_variable_values, str):
            missing_variable_fill_function = lambda var_name: missing_variable_values # use the literal string itself
        elif isinstance(missing_variable_values, dict):
            missing_variable_fill_function = lambda var_name: missing_variable_values.get(var_name, 'None') # find the value in the dictionary, or use 'None'
        else:
            raise NotImplementedError



        while (num_iterations <= max_iterations_before_abort) and ((target_dict is None) or (not isinstance(target_dict, dict))):
            try:
                # Tries to evaluate the code_dict_defn, which is just a string, into a valid dictionary object
                target_dict = eval(code_dict_defn) # , None, None # should produce NameError: name 'curr_ax_firing_rate' is not defined
            except NameError as e:
                if debug_print:
                    print(f'iteration {num_iterations}: {e}')
                last_exception = e
                # when e is a NameError, str(e) is a stirng like: "name 'curr_ax_firing_rate' is not defined"
                missing_variable_name = missing_variable_name = cls._parse_NameError(e) # e.g. 'curr_ax_firing_rate'
                if debug_print:
                    print(f'missing_variable_name: {missing_variable_name}')

                # missing_variable_assignment_str = f'{missing_variable_name} = None' # old way, always fill with None
                missing_variable_assignment_str = f'{missing_variable_name} = {missing_variable_fill_function(missing_variable_name)}'
                exec(missing_variable_assignment_str) # define the missing variable as None in this environment
            except Exception as e:
                # Unhandled/Unexpected exception:
                print(f'ERROR: iteration {num_iterations}: Unhandled exception: {e}')
                last_exception = e
                raise e
            num_iterations = num_iterations + 1

        if target_dict is None:
            # still no resolved dict, failed
            if last_exception is not None:
                raise last_exception
            else:
                raise NotImplementedError
        else:
            return target_dict


    @classmethod
    def convert_defn_lines_to_dictionary(cls, code, multiline_dict_defn=True, multiline_members_indent='    '):
        """ Converts a multiline string containing lines of valid python code definitions into an output string containing a python dictionary definition.

            code: lines of code that define several python variables to be converted to dictionary entries
            multiline_dict_defn: if True, each entry is converted to a new line (multi-line dict defn). Otherwise inline dict defn.

        Implementation: Internally calls cls._convert_defn_line_to_dictionary_line(...) for each line

        Examples:
            test_parameters_defns_code_string = '''
                max_num_spikes_per_neuron = 20000 # the number of spikes to truncate each neuron's timeseries to
                kleinberg_parameters = DynamicParameters(s=2, gamma=0.1)
                use_progress_bar = False # whether to use a tqdm progress bar
                debug_print = False # whether to print debug-level progress using traditional print(...) statements
            '''
            >>> "\nmax_num_spikes_per_neuron = 20000 # the number of spikes to truncate each neuron's timeseries to\nkleinberg_parameters = DynamicParameters(s=2, gamma=0.1)\nuse_progress_bar = False # whether to use a tqdm progress bar\ndebug_print = False # whether to print debug-level progress using traditional print(...) statements\n"

            active_str = CodeConversion.convert_defn_lines_to_dictionary(test_parameters_defns_code_string, multiline_dict_defn=False)
            active_str
            >>> "{'max_num_spikes_per_neuron': 20000, 'kleinberg_parameters': DynamicParameters(s=2, gamma=0.1), 'use_progress_bar': False, 'debug_print': False}"

            print(convert_defn_lines_to_dictionary(test_parameters_defns_code_string, multiline_dict_defn=True))
            >>>
                {
                'max_num_spikes_per_neuron': 20000,
                'kleinberg_parameters': DynamicParameters(s=2, gamma=0.1),
                'use_progress_bar': False,
                'debug_print': False
                }
        """
        code = str(code)
        code_lines = code.splitlines() # assumes one definition per line
        formatted_code_lines = []

        for i in np.arange(len(code_lines)):
            # Remove any trailing comments:
            curr_code_line = code_lines[i]
            curr_code_line = curr_code_line.strip() # strip leading and trailing whitespace
            if len(curr_code_line) > 0:
                curr_code_line = cls._stripComments(curr_code_line).strip()
                curr_code_line = cls._convert_defn_line_to_dictionary_line(curr_code_line).strip()
                if multiline_dict_defn:
                    # if multiline dict, indent the entries by the specified amount
                    curr_code_line = f'{multiline_members_indent}{curr_code_line}'

                formatted_code_lines.append(curr_code_line)
        # formatted_code_lines
        # Build final flattened output string:
        if multiline_dict_defn:
            dict_entry_seprator=',\n'
            dict_prefix = '{\n'
            dict_suffix = '\n}\n'
        else:
            dict_entry_seprator=', '
            dict_prefix = '{'
            dict_suffix = '}'

        flat_dict_member_code_str = dict_entry_seprator.join(formatted_code_lines)
        final_dict_defn_str = dict_prefix + flat_dict_member_code_str + dict_suffix
        return final_dict_defn_str


    @classmethod
    def convert_defn_lines_to_parameters_list(cls, code):
        """
        Inputs:
            code: lines of code that define several python variables to be converted to parameters, as would be passed into a function


        Examples:
            test_parameters_defns_code_string = '''
                max_num_spikes_per_neuron = 20000 # the number of spikes to truncate each neuron's timeseries to
                kleinberg_parameters = DynamicParameters(s=2, gamma=0.1)
                use_progress_bar = False # whether to use a tqdm progress bar
                debug_print = False # whether to print debug-level progress using traditional print(...) statements
            '''

            CodeConversion.convert_defn_lines_to_parameters_list(test_parameters_defns_code_string)
            >>> 'max_num_spikes_per_neuron=20000, kleinberg_parameters=DynamicParameters(s=2, gamma=0.1), use_progress_bar=False, debug_print=False'

        """
        code = str(code)
        code_lines = code.splitlines() # assumes one definition per line
        formatted_code_lines = []

        for i in np.arange(len(code_lines)):
            # Remove any trailing comments:
            curr_code_line = code_lines[i]
            curr_code_line = curr_code_line.strip() # strip leading and trailing whitespace
            if len(curr_code_line) > 0:
                curr_code_line = cls._stripComments(curr_code_line).strip()
                curr_code_line = cls._convert_defn_line_to_parameter_list_item(curr_code_line).strip()
                formatted_code_lines.append(curr_code_line)
        # formatted_code_lines
        # Build final flattened output string:
        item_entry_seprator=', '
        list_prefix = '' # '['
        list_suffix = '' # ']'
        flat_list_member_code_str = item_entry_seprator.join(formatted_code_lines)
        final_list_defn_str = list_prefix + flat_list_member_code_str + list_suffix
        return final_list_defn_str

    @classmethod
    def convert_variable_tuple_code_to_dict_with_names(cls, tuple_string: str) -> str:
        """ Given a line of code representing a simple tuple or comma separated list of variable names (such as would be returned from a function that returns multiple outputs, or placed on the LHS of a multi-item assignment) returns a transformed line of code representing a dictionary with keys equal to the variable name and value equal to the variable value.

        Example:
            from pyphocorehelpers.general_helpers import CodeConversion

            tuple_string = "(active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean)"
            result_dict_str_rep, result_dict = CodeConversion.convert_variable_tuple_code_to_dict_with_names(tuple_string)
            print(result_dict_str_rep)
            >>> {'active_filter_epochs':active_filter_epochs, 'original_1D_decoder':original_1D_decoder, 'all_included_filter_epochs_decoder_result':all_included_filter_epochs_decoder_result, 'flat_all_epochs_measured_cell_spike_counts':flat_all_epochs_measured_cell_spike_counts, 'flat_all_epochs_measured_cell_firing_rates':flat_all_epochs_measured_cell_firing_rates, 'flat_all_epochs_decoded_epoch_time_bins':flat_all_epochs_decoded_epoch_time_bins, 'flat_all_epochs_computed_surprises':flat_all_epochs_computed_surprises, 'flat_all_epochs_computed_expected_cell_firing_rates':flat_all_epochs_computed_expected_cell_firing_rates, 'flat_all_epochs_difference_from_expected_cell_spike_counts':flat_all_epochs_difference_from_expected_cell_spike_counts, 'flat_all_epochs_difference_from_expected_cell_firing_rates':flat_all_epochs_difference_from_expected_cell_firing_rates, 'all_epochs_decoded_epoch_time_bins_mean':all_epochs_decoded_epoch_time_bins_mean, 'all_epochs_computed_cell_surprises_mean':all_epochs_computed_cell_surprises_mean, 'all_epochs_all_cells_computed_surprises_mean':all_epochs_all_cells_computed_surprises_mean}

        """
        # Remove the parentheses from the string
        tuple_string = tuple_string.strip('()')

        # Split the string into a list of variable names
        variable_names = tuple_string.split(',')

        # Create a dictionary with variable names as keys and None as values
        result_dict = {f"'{name.strip()}'": name.strip() for name in variable_names}
        result_dict_str_rep = ', '.join([f"'{name.strip()}':{name.strip()}" for name in variable_names])
        result_dict_str_rep = '{' + result_dict_str_rep + '}'
        return result_dict_str_rep, result_dict

    # Static Helpers: ____________________________________________________________________________________________________ #
    @classmethod
    def get_arguments_as_optional_dict(cls, *args, **kwargs):
        """ Easily converts your existing argument-list style default values into a dict:
                Defines a simple function that takes only **kwargs as its inputs and prints the values it recieves. Paste your values as arguments to the function call. The dictionary will be output to the console, so you can easily copy and paste.
            Usage:
                >>> get_arguments_as_optional_dict(point_size=8, font_size=10, name='build_center_labels_test', shape_opacity=0.8, show_points=False)

                Output: ", **({'point_size': 8, 'font_size': 10, 'name': 'build_center_labels_test', 'shape_opacity': 0.8, 'show_points': False} | kwargs)"

            Usage (string-represented kwargs mode):
                Consider in code:
                    `sortby=shared_fragile_neuron_IDXs, included_unit_neuron_IDs=curr_any_context_neurons, fignum=None, ax=ax_long_pf_1D, curve_hatch_style=None`

                We'd like to convert this to an optional dict, which can usually be done by passing it to CodeConversion.get_arguments_as_optional_dict(...) like:
                    `CodeConversion.get_arguments_as_optional_dict(sortby=shared_fragile_neuron_IDXs, included_unit_neuron_IDs=curr_any_context_neurons, fignum=None, ax=ax_long_pf_1D, curve_hatch_style=None)`

                Unfortunately, unless done in the original calling context many of the arguments are undefined, including:
                    shared_fragile_neuron_IDXs, curr_any_context_neurons, ax_long_pf_1D

import pyphocorehelpers.programming_helpers                >>> pyphocorehelpers.programming_helpers.get_arguments_as_optional_dict("sortby=shared_fragile_neuron_IDXs, included_unit_neuron_IDs=curr_any_context_neurons, ax=ax_long_pf_1D", fignum=None, curve_hatch_style=None)

                Output: , **({'fignum': None, 'curve_hatch_style': None, 'sortby': shared_fragile_neuron_IDXs, 'included_unit_neuron_IDs': curr_any_context_neurons, 'ax': ax_long_pf_1D} | kwargs)
        """
        if len(args) == 0:
            # default mode, length of arguments is zero
            replacement_wrapped_undefined_variable_dict = {} # TODO: unknown if right, but works around undefined `replacement_wrapped_undefined_variable_dict` when there are only kwargs
        else:
            # check for string input mode:
            assert len(args) == 1, f"only string-represented kwargs are allowed as a non-keyword argument, but args: {args} (with length {len(args)} instead of 1) were passed."
            str_rep_kwargs = args[0]
            assert isinstance(str_rep_kwargs, str)
            ## Try and parse the kwargs to a valid kwargs dict, ignoring NameErrors using
            code_dict_defn=f"dict({str_rep_kwargs})"
            try:
                undefined_variable_names = CodeConversion.extract_undefined_variable_names_from_code(code_dict_defn)
                replacement_wrapped_undefined_variable_dict = {a_name:f"'`{a_name}`'" for a_name in undefined_variable_names} # wrap each variable in markdown-style code quotes and then single quotes
                parsed_kwargs = cls.build_dummy_dictionary_from_defn_code(code_dict_defn=code_dict_defn, max_iterations_before_abort=50, missing_variable_values=replacement_wrapped_undefined_variable_dict, debug_print=True)
                kwargs = kwargs | parsed_kwargs # Combine the parsed_kwargs and the correctly passed kwargs into a combined dictionary to be used.
            except Exception as e:
                print(f'Interpreting as string-representation arg-list and converting to a dictionary resulted in code_dict_defn: {code_dict_defn} but still failed! Exception: {e}')
                raise e

        out_dict_str = f'{kwargs}'
        ## replace the sentinal wrapped values once the dict is built
        for orig_name, sentinal_wrapped_name in replacement_wrapped_undefined_variable_dict.items():
            out_dict_str = out_dict_str.replace(sentinal_wrapped_name, orig_name) # restore the non-sentinal-wrapped variable names that were subsituted in

        print(', **(' + f'{out_dict_str}' + ' | kwargs)')


    @classmethod
    def _build_class_init_fcn(cls, target_dict, class_name:str='ClassName', output_variable_prefix='', include_type_hinting=False, include_default_values=False,
        indent_character='    '):
        """

        Usage:

        from pyphocorehelpers.general_helpers import CodeConversion
        init_fcn_code_str = CodeConversion._build_class_init_fcn(_out.to_dict(), class_name='PlacefieldSnapshot', include_type_hinting=False, include_default_values=False)
        init_fcn_code_str

        >>> Output:
        def __init__(self, num_position_samples_occupancy, seconds_occupancy, spikes_maps_matrix, smoothed_spikes_maps_matrix, occupancy_weighted_tuning_maps_matrix):
            super(PlacefieldSnapshot, self).__init__()        self.num_position_samples_occupancy = num_position_samples_occupancy
            self.seconds_occupancy = seconds_occupancy
            self.spikes_maps_matrix = spikes_maps_matrix
            self.smoothed_spikes_maps_matrix = smoothed_spikes_maps_matrix
            self.occupancy_weighted_tuning_maps_matrix = occupancy_weighted_tuning_maps_matrix


        """
        init_final_variable_names_list = [f"{output_variable_prefix}{k}" for k,v in target_dict.items()]


        if include_type_hinting:
            # by default type(v) gives <class 'numpy.ndarray'>
            init_arguments_list = [f"{output_variable_prefix}{k}: {cls._find_best_type_representation_string(type(v))}" for k,v in target_dict.items()]
        else:
            # no type hinting:
            init_arguments_list = init_final_variable_names_list.copy()

        if include_default_values:
            init_arguments_list = [f"{curr_arg_str}={v}" for curr_arg_str, v in zip(init_arguments_list, target_dict.values())]

        member_properties_code_str = ', '.join(init_arguments_list)
        member_assignments_code_str = '\n'.join([f'{indent_character}{indent_character}self.{a_final_arg_name} = {a_final_arg_name}' for a_final_arg_name in init_final_variable_names_list])

        return f"""{indent_character}def __init__(self, {member_properties_code_str}):\n{indent_character}{indent_character}super({class_name}, self).__init__(){member_assignments_code_str}"""


    @classmethod
    def generate_unwrap_code_from_dict_like(cls, **kwargs) -> Tuple[str, List[str]]:
        """ Generate unwrapping code from a dict-like class
        Usage:
            from pyphocorehelpers.programming_helpers import CodeConversion

            code_lines_str, code_lines = CodeConversion.generate_unwrap_code_from_dict_like(short_long_pf_overlap_analyses=short_long_pf_overlap_analyses)
            code_lines_str
        

        >> Output:
            ## Unwrapping `short_long_pf_overlap_analyses`:
            short_long_neurons_diff = short_long_pf_overlap_analyses['short_long_neurons_diff']
            poly_overlap_df = short_long_pf_overlap_analyses['poly_overlap_df']
            conv_overlap_dict = short_long_pf_overlap_analyses['conv_overlap_dict']
            conv_overlap_scalars_df = short_long_pf_overlap_analyses['conv_overlap_scalars_df']
            product_overlap_dict = short_long_pf_overlap_analyses['product_overlap_dict']
            product_overlap_scalars_df = short_long_pf_overlap_analyses['product_overlap_scalars_df']
            relative_entropy_overlap_dict = short_long_pf_overlap_analyses['relative_entropy_overlap_dict']
            relative_entropy_overlap_scalars_df = short_long_pf_overlap_analyses['relative_entropy_overlap_scalars_df']


        """
        # short_long_pf_overlap_analyses
        # a_dict_like.keys()
        # a_dict_like
        assert len(kwargs) == 1, f"please pass kwargs one at at time, like generate_unwrap_code(short_long_pf_overlap_analyses=short_long_pf_overlap_analyses)"
        dict_like_name: str = str(list(kwargs.keys())[0]) # 'short_long_pf_overlap_analyses'
        a_dict_like = list(kwargs.values())[0]
        code_lines = []
        for k, v in a_dict_like.items():
            a_code_line: str = f"{k} = {dict_like_name}['{k}']"
            code_lines.append(a_code_line)
            
        code_lines_str: str = '\n'.join(code_lines)
        print(f'## Unwrapping `{dict_like_name}`:\n{code_lines_str}\n\n')
        return code_lines_str, code_lines




@function_attributes(short_name=None, tags=['attrs', 'class', 'make_class', 'dict'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-15 00:00', related_items=[])
@classmethod
def create_class_from_dict(cls, class_name, input_dict):
    """ Programmatic Attr Class Generation with attr.ib     
    TempGraphicsOutput = create_class_from_dict('TempGraphicsOutput', _out)
    TempGraphicsOutput
    """
    import attr
    import attrs

    attributes = {}
    for key, value in input_dict.items():
        attributes[key] = attr.ib(type=type(value), default=value) # , repr=False

    return attrs.make_class(class_name, attributes)

