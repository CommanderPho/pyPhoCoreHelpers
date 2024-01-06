# title: programming_helpers.py
# date: 2023-05-08 14:21:48
# purpose: Created to support programming and consolidation of programming-related helpers into a single location. Previously all were scattered around the various other helpers.
import os
import sys
import contextlib
from pathlib import Path
from typing import Optional, List, Dict, Union
from functools import wraps
import numpy as np
import pandas as pd
import inspect # for IPythonHelpers
from enum import Enum
import re
import ast
import nbformat
import IPython
from IPython.display import display, Javascript
import json
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
    df = pd.DataFrame([code_str])
    df.to_clipboard(index=False,header=False)
    if message_print:
        print(f'Copied "{code_str}" to clipboard!')
    
            
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