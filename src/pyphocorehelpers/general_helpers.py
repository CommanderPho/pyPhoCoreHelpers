from typing import Callable, List, Optional, OrderedDict  # for OrderedMeta
from enum import Enum
import re # for CodeConversion
import numpy as np # for CodeConversion


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



def inspect_callable_arguments(a_callable: Callable):
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
    fn_spec = inspect.getfullargspec(a_callable)
    # fn_sig = inspect.signature(compute_position_grid_bin_size)
    return fn_spec


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
    def convert_defn_lines_to_dictionary(cls, code, multiline_dict_defn=True, multiline_members_indent='    '):
        """ 
            code: lines of code that define several python variables to be converted to dictionary entries
            multiline_dict_defn: if True, each entry is converted to a new line (multi-line dict defn). Otherwise inline dict defn.
            
            
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

   
