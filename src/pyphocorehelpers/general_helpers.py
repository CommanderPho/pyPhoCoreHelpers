from typing import Callable, List, Optional, OrderedDict  # for OrderedMeta
from enum import Enum
import re # for CodeConversion
import numpy as np # for CodeConversion
import pandas as pd
from neuropy.utils.dynamic_container import overriding_dict_with # required for safely_accepts_kwargs


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



def inspect_callable_arguments(a_callable: Callable, debug_print=False):
    """ Not yet validated/implemented
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

    return full_fn_spec, positional_args_names, kwargs_names, default_kwargs_dict

def safely_accepts_kwargs(fn):
    """ builds a wrapped version of fn that only takes the kwargs that it can use, and shrugs the rest off 
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




class CodeConversion(object):
    """ Converts code (usually passed as text) to various alternative formats to ease development workflows. 
    
    
    """
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
    def convert_dictionary_to_defn_lines(cls, target_dict, multiline_assignment_code=False, dictionary_name:str='target_dict', include_comment:bool=True, copy_to_clipboard=True):
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
        if isinstance(target_dict, str):
            # if the target_dict is a string instead of a dictionary, assume it is code that defines a dictionary
            try:
                target_dict = cls.build_dummy_dictionary_from_defn_code(target_dict)
            except Exception as e:
                print(f'ERROR: Could not convert code string: {target_dict} to a proper dictionary! Exception: {e}')
                raise e

        assert isinstance(target_dict, dict), f"target_dict must be a dictionary but is of type: {type(target_dict)}, target_dict: {target_dict}"
        comment_str = f"# Extract variables from the `{dictionary_name}` dictionary to the local workspace"
        if multiline_assignment_code:
            # Separate line per assignment
            """
            # Extract variables from the `{dictionary_name}` dictionary to the local workspace
            spike_raster_plt_2d = target_dict['spike_raster_plt_2d']
            spike_raster_plt_3d = target_dict['spike_raster_plt_3d']
            spike_raster_window = target_dict['spike_raster_window']

            """
            code_str = '\n'.join([f"{k} = {dictionary_name}['{k}']" for k,v in target_dict.items()])

            if include_comment:
                code_str = f"{comment_str}\n{code_str}" # add comment above code

        else:
            # Generates an inline assignment, e.g. "spike_raster_plt_2d, spike_raster_plt_3d, spike_raster_window = target_dict['spike_raster_plt_2d'], target_dict['spike_raster_plt_3d'], target_dict['spike_raster_window']"
            """ Assignment all on a single line
            spike_raster_plt_2d, spike_raster_plt_3d, spike_raster_window = target_dict['spike_raster_plt_2d'], target_dict['spike_raster_plt_3d'], target_dict['spike_raster_window'] # Extract variables from the `target_dict` dictionary to the local workspace
            """
            vars_name_list = list(target_dict.keys())
            rhs_code_str = ', '.join([f"{dictionary_name}['{k}']" for k in vars_name_list])
            lhs_code_str = ', '.join(vars_name_list)
            code_str = f'{lhs_code_str} = {rhs_code_str}'
            if include_comment:
                code_str = f"{code_str} {comment_str}" # add comment at the very end of the code line

        if copy_to_clipboard:
            df = pd.DataFrame([code_str])
            df.to_clipboard(index=False,header=False)
            print(f'Copied "{code_str}" to clipboard!')

        return code_str


    @classmethod
    def build_dummy_dictionary_from_defn_code(cls, code_dict_defn:str, max_iterations_before_abort:int=50, debug_print=False):
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
        while (num_iterations <= max_iterations_before_abort) and ((target_dict is None) or (not isinstance(target_dict, dict))):
            try:
                # Tries to turn the code_dict_defn, which is just a string, into a valid dictionary object
                target_dict = eval(code_dict_defn) # , None, None # should produce NameError: name 'curr_ax_firing_rate' is not defined
            except NameError as e:
                if debug_print:
                    print(f'iteration {num_iterations}: {e}')
                last_exception = e
                # when e is a NameError, str(e) is a stirng like: "name 'curr_ax_firing_rate' is not defined"
                name_error_str = str(e)
                name_error_split_str = name_error_str.split("'")
                assert len(name_error_split_str)==3, f"name_error_split_str: {name_error_split_str}"
                missing_variable_name = name_error_split_str[1] # e.g. 'curr_ax_firing_rate'
                if debug_print:
                    print(f'missing_variable_name: {missing_variable_name}')
                exec(f'{missing_variable_name} = None') # define the missing variable as None
                # print(f'\te.name: {e.name}')
            except Exception as e:
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
        """ 
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
    
    

    ## Static Helpers:
    @classmethod
    def get_arguments_as_optional_dict(cls, **kwargs):
        """ Easily converts your existing argument-list style default values into a dict:
                Defines a simple function that takes only **kwargs as its inputs and prints the values it recieves. Paste your values as arguments to the function call. The dictionary will be output to the console, so you can easily copy and paste. 
            Usage:
                >>> get_arguments_as_optional_dict(point_size=8, font_size=10, name='build_center_labels_test', shape_opacity=0.8, show_points=False)

                Output: ", **({'point_size': 8, 'font_size': 10, 'name': 'build_center_labels_test', 'shape_opacity': 0.8, 'show_points': False} | kwargs)"
        """
        print(', **(' + f'{kwargs}' + ' | kwargs)')



# Enum for size units
class SIZE_UNIT(Enum):
    BYTES = 1
    KB = 2
    MB = 3
    GB = 4
   
def convert_unit(size_in_bytes, unit):
    """ Convert the size from bytes to other units like KB, MB or GB"""
    if unit == SIZE_UNIT.KB:
        return size_in_bytes/1024
    elif unit == SIZE_UNIT.MB:
        return size_in_bytes/(1024*1024)
    elif unit == SIZE_UNIT.GB:
        return size_in_bytes/(1024*1024*1024)
    else:
        return size_in_bytes

   
