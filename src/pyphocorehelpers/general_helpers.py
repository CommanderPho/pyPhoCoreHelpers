from contextlib import contextmanager
from typing import Callable, List, Dict, Tuple, Optional, OrderedDict  # for OrderedMeta
from collections import namedtuple
import sys # needed for `is_reloaded_instance`
from enum import Enum
from enum import unique # GeneratedClassDefinitionType
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum # required for GeneratedClassDefinitionType

import re # for CodeConversion
import numpy as np # for CodeConversion
import pandas as pd
from neuropy.utils.dynamic_container import overriding_dict_with # required for safely_accepts_kwargs


"""

## Pho Programming Helpers:
import inspect
from pyphocorehelpers.general_helpers import inspect_callable_arguments, get_arguments_as_optional_dict, GeneratedClassDefinitionType, CodeConversion
from pyphocorehelpers.print_helpers import DocumentationFilePrinter, TypePrintMode, print_keys_if_possible, debug_dump_object_member_shapes, print_value_overview_only, document_active_variables, CapturedException

"""

class OrderedMeta(type):
    """Replaces the inheriting object's dict of attributes with an OrderedDict that preserves enumeration order
    Reference: https://stackoverflow.com/questions/11296010/iterate-through-class-members-in-order-of-their-declaration
    Usage:
        # Set the metaclass property of your custom class to OrderedMeta
        class Person(metaclass=OrderedMeta):
            name = None
            date_of_birth = None
            nationality = None
            gender = None
            address = None
            comment = None

        # Can then enumerate members while preserving order
        for member in Person._orderedKeys:
            if not getattr(Person, member):
                print(member)
    """

    @classmethod
    def __prepare__(metacls, name, bases):
        return OrderedDict()

    def __new__(cls, name, bases, clsdict):
        c = type.__new__(cls, name, bases, clsdict)
        c._orderedKeys = clsdict.keys()
        return c


FunctionInspectionTuple = namedtuple('FunctionInspectionTuple', ['full_fn_spec', 'positional_args_names', 'kwargs_names', 'default_kwargs_dict'])

def inspect_callable_arguments(a_callable: Callable, debug_print=False) -> FunctionInspectionTuple:
    """ Inspects a callable's arguments
    Progress:
        import inspect
        from neuropy.plotting.ratemaps import plot_ratemap_1D, plot_ratemap_2D

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


def safely_accepts_kwargs(fn):
    """ builds a wrapped version of fn that only takes the kwargs that it can use, and shrugs the rest off (without any warning that they're unused, making it a bit dangerous)
    Can be used as a decorator to make any function gracefully accept unhandled kwargs

    Can be used to conceptually "splat" a configuration dictionary of properties against a function that only uses a subset of them, such as might need to be done for plotting, etc)
    
    Usage:
        @safely_accepts_kwargs
        def _test_fn_with_limited_parameters(item1=None, item2='', item3=5.0):
            print(f'item1={item1}, item2={item2}, item3={item3}')
            
            
    TODO: Tests:
        from pyphocorehelpers.general_helpers import safely_accepts_kwargs

        # def _test_fn_with_limited_parameters(newitem, item1=None, item2='', item3=5.0):
        #     print(f'item1={item1}, item2={item2}, item3={item3}')

        @safely_accepts_kwargs
        def _test_fn_with_limited_parameters(item1=None, item2='', item3=5.0):
            print(f'item1={item1}, item2={item2}, item3={item3}')

        @safely_accepts_kwargs
        def _test_fn2_with_limited_parameters(itemA=None, itemB='', itemC=5.0):
            print(f'itemA={itemA}, itemB={itemB}, itemC={itemC}')
            
        def _test_outer_fn(**kwargs):
            _test_fn_with_limited_parameters(**kwargs)
            _test_fn2_with_limited_parameters(**kwargs)
            # _test_fn_with_limited_parameters(**overriding_dict_with(lhs_dict=fn_spec_default_arg_dict, **kwargs))
            # _test_fn2_with_limited_parameters(**overriding_dict_with(lhs_dict=fn_spec_default_arg_dict, **kwargs))
            
            # Build safe versions of the functions
            # _safe_test_fn_with_limited_parameters = _build_safe_kwargs(_test_fn_with_limited_parameters)
            # _safe_test_fn2_with_limited_parameters = _build_safe_kwargs(_test_fn2_with_limited_parameters)
            # Call the safe versions:
            # _safe_test_fn_with_limited_parameters(**kwargs)
            # _safe_test_fn2_with_limited_parameters(**kwargs)
            
            
        # _test_outer_fn()
        _test_outer_fn(itemB=15) # TypeError: _test_fn_with_limited_parameters() got an unexpected keyword argument 'itemB'

    """
    full_fn_spec, positional_args_names, kwargs_names, default_kwargs_dict = inspect_callable_arguments(fn)
    def _safe_kwargs_fn(*args, **kwargs):
        return fn(*args, **overriding_dict_with(lhs_dict=default_kwargs_dict, **kwargs))
    return _safe_kwargs_fn



# def get_arguments_as_passthrough(**kwargs):
    

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



class CodeConversion(object):
    """ Converts code (usually passed as text) to various alternative formats to ease development workflows. 


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
    _types_replace_dict = {'numpy.':'np.', 'pandas.':'pd.'}
    _inverse_types_replace_dict = {v:k for k,v in _types_replace_dict.items()}


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

        assert isinstance(target_dict, dict), f"target_dict must be a dictionary but is of type: {type(target_dict)}, target_dict: {target_dict}"
        return target_dict # returns a for-sure dictionary or throws an Exception
        
    @classmethod
    def _find_best_type_representation_string(cls, a_type, unspecified_generic_type_name='type', keep_generic_types=['NoneType'], types_replace_dict = {'numpy.':'np.', 'pandas.':'pd.'}):
        """ Uses `strip_type_str_to_classname(a_type) to find the best type-string representation.

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
                    full_types_str_dict = {k:f"{cls._find_best_type_representation_string(type(v))}" for k,v in target_dict.items()}
                    
                    relative_types_dict = {}
                    for k, full_type_string in full_types_str_dict.items():
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
                            relative_types_dict[k] = full_type_string
                            import_statement = None # f'import {split_type_components[0]}' # 'import numpy as np' TODO: import numpy/pd
                        else:
                            relative_types_dict[k] = class_name

                        if (import_statement is not None) and (import_statement not in needed_import_statements):
                            needed_import_statements.append(import_statement)


                    # Apply the find/replace dict to fix issues like '' being output 
                    relative_types_dict = {k:cls.apply_find_replace(find_replace_dict=cls._general_find_replace_dict, target_str=v) for k,v in relative_types_dict.items()}
                    
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

            active_str = convert_defn_lines_to_dictionary(test_parameters_defns_code_string, multiline_dict_defn=False)
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

                >>> CodeConversion.get_arguments_as_optional_dict("sortby=shared_fragile_neuron_IDXs, included_unit_neuron_IDs=curr_any_context_neurons, ax=ax_long_pf_1D", fignum=None, curve_hatch_style=None)

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


# Enum for size units
class SIZE_UNIT(Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4
   
def convert_unit(size_in_bytes, unit: SIZE_UNIT):
    """ Convert the size from bytes to other units like KB, MB or GB"""
    if unit == SIZE_UNIT.KB:
        return size_in_bytes/1024
    elif unit == SIZE_UNIT.MB:
        return size_in_bytes/(1024*1024)
    elif unit == SIZE_UNIT.GB:
        return size_in_bytes/(1024*1024*1024)
    else:
        return size_in_bytes

   
# @metadata_attributes(short_name=None, tags=['contextmanager'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-07-05 19:02', related_items=[])
@contextmanager
def disable_function_context(obj, fn_name: str):
    """ Disables a function within a context manager


    https://stackoverflow.com/questions/10388411/possible-to-globally-replace-a-function-with-a-context-manager-in-python

    Could be used for plt.show().
    ```python
    
    from pyphocorehelpers.general_helpers import disable_function_context
    import matplotlib.pyplot as plt
    with disable_function_context(plt, "show"):
        run_me(x)
    ```

    

    """
    temp = getattr(obj, fn_name)
    setattr(obj, fn_name, lambda: None)
    yield
    setattr(obj, fn_name, temp)
    


# from dataclasses import dataclass


# @dataclass
# class LoadBuildSave(object):
#     """TODO 2023-04-11 - UNFINISHED 
#         Tries to load the object from file first
#         If this isn't possible it runs something to compute it
#         Then it saves it so it can be loaded in the future.

#         Combines the code to perform this common procedure in a concise structure instead of requiring it to be spread out over several places.


#     Example 0: from `pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing.batch_extended_computations`

# ```python
#         try:
#             ## Get global 'jonathan_firing_rate_analysis' results:
#             curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
#             neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis.neuron_replay_stats_df, curr_jonathan_firing_rate_analysis.rdf.rdf, curr_jonathan_firing_rate_analysis.rdf.aclu_to_idx, curr_jonathan_firing_rate_analysis.irdf.irdf
#             if progress_print:
#                 print(f'{_comp_name} already computed.')
#         except (AttributeError, KeyError) as e:
#             if progress_print or debug_print:
#                 print(f'{_comp_name} missing.')
#             if debug_print:
#                 print(f'\t encountered error: {e}\n{traceback.format_exc()}\n.')
#             if progress_print or debug_print:
#                 print(f'\t Recomputing {_comp_name}...')
#             curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_jonathan_replay_firing_rate_analyses'], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
#             print(f'\t done.')
#             curr_jonathan_firing_rate_analysis = curr_active_pipeline.global_computation_results.computed_data['jonathan_firing_rate_analysis']
#             neuron_replay_stats_df, rdf, aclu_to_idx, irdf = curr_jonathan_firing_rate_analysis.neuron_replay_stats_df, curr_jonathan_firing_rate_analysis.rdf.rdf, curr_jonathan_firing_rate_analysis.rdf.aclu_to_idx, curr_jonathan_firing_rate_analysis.irdf.irdf
#             newly_computed_values.append(_comp_name)
#         except Exception as e:
#             raise e


#     Example 1: from `pyphoplacecellanalysis.Analysis.Decoder.decoder_result.perform_full_session_leave_one_out_decoding_analysis`
#         # Save to file to cache in case we crash:
#         leave_one_out_surprise_result_pickle_path = output_data_folder.joinpath(f'leave_one_out_surprise_results{cache_suffix}.pkl').resolve()
#         print(f'leave_one_out_surprise_result_pickle_path: {leave_one_out_surprise_result_pickle_path}')
#         saveData(leave_one_out_surprise_result_pickle_path, (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, 
#                                                             flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, 
#                                                             flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates,
#                                                             flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates,
#                                                             all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean))


# ```

#     """
#     property: type
    


# ==================================================================================================================== #
# UNTESTED                                                                                                             #
# ==================================================================================================================== #

# def is_reloaded_instance(obj, classinfo):
#     """ determines if a class instance is a reloaded instance of a class"""
#     return isinstance(obj, classinfo) and sys.getrefcount(classinfo) > 1



# def get_regular_attrs(obj, include_parent=True):
#     """ Intended to get all of the stored attributes of an object, including those inherited from parent classes, while ignoring @properties and other computed variables
#     Example:
#         class ParentClass:
#             def __init__(self, z):
#                 self.z = z

#         class MyClass(ParentClass):
#             def __init__(self, x):
#                 super().__init__(x+1)
#                 self.x = x
#                 self.y = x + 1

#             @property
#             def computed_prop(self):
#                 return self.x + self.y

#         obj = MyClass(5)
#         regular_attrs = get_regular_attrs(obj)
#         print(regular_attrs)  # Output: ['z', 'x', 'y']
    

#     ISSUE: returns propery when defined this way

#     @property
#     def pdf_normalized_tuning_curves(self):
#         return Ratemap.perform_AOC_normalization(self.tuning_curves)


#     Usage:
#         get_regular_attrs(ratemap_2D, include_parent=False)

#     """
#     regular_attrs = []
#     cls = type(obj)
#     while cls:
#         for attr in cls.__dict__:
#             if not callable(getattr(obj, attr)) and not attr.startswith('__'):
#                 regular_attrs.append(attr)
#         if not include_parent:
#             break
#         cls = cls.__base__
#     return list(set(regular_attrs))



