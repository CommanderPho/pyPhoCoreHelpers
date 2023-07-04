# title: programming_helpers.py
# date: 2023-05-08 14:21:48
# purpose: Created to support programming and consolidation of programming-related helpers into a single location. Previously all were scattered around the various other helpers.


from functools import wraps
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


import inspect

class IPythonHelpers:
    """ various helpers useful in jupyter-lab notebooks and IPython """
    
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




from enum import Enum
import re


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






