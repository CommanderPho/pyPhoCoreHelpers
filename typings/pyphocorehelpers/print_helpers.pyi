"""
This type stub file was generated by pyright.
"""

import pandas as pd
import logging
from typing import Any, List, Optional, Union
from nptyping import NDArray
from IPython.display import HTML
from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum

def truncating_list_repr(items, max_full_display_n_items: int = ..., truncated_num_edge_items: int = ...): # -> str:
    """ If length is less than `max_full_display_n_items` return the full list 
    https://stackoverflow.com/questions/62884503/what-are-the-best-practices-for-repr-with-collection-class-python
    

    Usage:
        from pyphocorehelpers.print_helpers import truncating_list_repr

        short_list = [1, 2, 3]
        medium_list = [1, 2, 3, 4, 5, 6]
        long_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(truncating_list_repr(short_list, max_full_display_n_items = 5, truncated_num_edge_items = 3)) # '[1, 2, 3]'
        print(truncating_list_repr(medium_list, max_full_display_n_items = 5, truncated_num_edge_items = 3)) # [1, 2, 3, ..., 4, 5, 6]
        truncating_list_repr(long_list, max_full_display_n_items = 5, truncated_num_edge_items = 3) # '[ 1,  2,  3, ...,  8,  9, 10]'


    """
    ...

class SimplePrintable:
    """Adds the default print method for classes that displays the class name and its dictionary.
    
    Shouldn't it define __str__(self) instead of __repr__(self)?
    """
    def __repr__(self) -> str:
        ...
    


class iPythonKeyCompletingMixin:
    """ Enables iPython key completion
    Requires Implementors to provide:
        self.keys()
    """
    ...


class PrettyPrintable(iPythonKeyCompletingMixin):
    def keys(self) -> List[Optional[str]]:
        ...
    


class WrappingMessagePrinter:
    """ 
    
    Examples:
        with WrappingMessagePrinter('Saving 2D Placefield image out to "{}"...'.format(active_plot_filepath), begin_line_ending='...', finished_message='done.'):
            for aFig in active_figures:
                aFig.savefig(active_plot_filepath)
    """
    def __init__(self, begin_string, begin_line_ending=..., finished_message=..., finished_line_ending=..., returns_string: bool = ..., enable_print: bool = ...) -> None:
        ...
    
    def __enter__(self): # -> None:
        ...
    
    def __exit__(self, *args): # -> None:
        ...
    
    @classmethod
    def print_generic_progress_message(cls, begin_string, begin_line_ending, returns_string, enable_print): # -> str | None:
        ...
    


class CustomTreeFormatters:
    @classmethod
    def basic_custom_tree_formatter(cls, depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=..., value_formatting_fn=...) -> str:
        """ For use with `print_keys_if_possible` to render a neat and pretty tree

            from pyphocorehelpers.print_helpers import CustomTreeFormatters

            
            print_keys_if_possible("sess.config.preprocessing_parameters", preprocessing_parameters_dict, custom_item_formatter=CustomTreeFormatters.basic_custom_tree_formatter)

        """
        ...
    


class ANSI_COLOR_STRINGS:
    """ Hardcoded ANSI-color strings. Can be used in print(...) like: `print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")` """
    HEADER = ...
    OKBLUE = ...
    OKCYAN = ...
    OKGREEN = ...
    WARNING = ...
    FAIL = ...
    RED = ...
    GREEN = ...
    YELLOW = ...
    BLUE = ...
    MAGENTA = ...
    CYAN = ...
    LIGHTRED = ...
    LIGHTGREEN = ...
    LIGHTYELLOW = ...
    LIGHTBLUE = ...
    LIGHTMAGENTA = ...
    LIGHTCYAN = ...
    ENDC = ...
    BOLD = ...
    UNDERLINE = ...


class ANSI_Coloring:
    """docstring for ANSI_Coloring."""
    def __init__(self, arg) -> None:
        ...
    
    @classmethod
    def ansi_highlight_only_suffix(cls, type_string, suffix_color=...): # -> str:
        """ From a FQDN-style type_string like 'pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult' generates a ansi-formatted string with the last suffix (the type name) colored. 
        Usage:
            type_string = 'pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult'
            ansi_highlighted_type_string = ansi_highlight_only_suffix(type_string)
            print(ansi_highlighted_type_string)
            >>> 'pyphoplacecellanalysis.General.Model.\x1b[93mComputationResult\x1b[0m'
        """
        ...
    


class DocumentationFilePrinter:
    """ Used to print and save readable data-structure documentation (in both plain and rich text) by wrapping `print_keys_if_possible(...)
    
        Usage:
            from pyphocorehelpers.print_helpers import DocumentationFilePrinter
            doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='ComputationResult')
            doc_printer.save_documentation('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'])

    """
    def __init__(self, doc_output_parent_folder=..., doc_name=..., custom_plain_text_formatter=..., custom_rich_text_formatter=..., enable_print: bool = ...) -> None:
        ...
    
    def save_documentation(self, *args, skip_save_to_file=..., skip_print=..., custom_plain_text_formatter=..., custom_rich_text_formatter=..., **kwargs): # -> None:
        """
            skip_print: if False, relies on self.enable_print's value to determine whether the output will be printed when this function is called
            
            Internally calls:
                print_keys_if_possible(*args, custom_rich_text_formatter=None, **kwargs) with custom_item_formatter= both plain and rich text formatters to print documentation
                saves to files unless skip_save_to_file=True
                
            Usage:
                doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='ComputationResult')
                doc_printer.save_documentation('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'])

        """
        ...
    
    def write_to_files(self): # -> None:
        """Write variables out to files"""
        ...
    
    def reveal_output_files_in_system_file_manager(self): # -> None:
        ...
    
    def display_widget(self): # -> VBox:
        """ Display an interactive jupyter-widget that allows you to open/reveal the generated files in the fileystem or default system display program. 
        """
        ...
    
    @classmethod
    def never_string_rep(cls, value_rep: str): # -> None:
        """ always returns None indicating no string-rep of the value should be included """
        ...
    
    @classmethod
    def string_rep_if_short_enough(cls, value: Any, max_length: int = ..., max_num_lines: int = ..., allow_reformatting: bool = ..., allow_ellipsis_fill_too_long_regions: bool = ..., debug_print: bool = ...): # -> str | None:
        """ returns the formatted str-rep of the value if it meets the criteria, otherwise nothing. An example `value_formatting_fn` 
        
        allow_reformatting: if True, allows removing lines to meet max_num_lines requirements so long as max_length is short enough
        
        
        Usage:
            from functools import partial
            from pyphocorehelpers.print_helpers import DocumentationFilePrinter

            custom_value_formatting_fn = partial(DocumentationFilePrinter.string_rep_if_short_enough, max_length=280, max_num_lines=1)
            new_custom_item_formatter = partial(DocumentationFilePrinter._default_rich_text_formatter, value_formatting_fn=custom_value_formatting_fn)
            print_keys_if_possible('context', context, max_depth=4, custom_item_formatter=new_custom_item_formatter)


        """
        ...
    
    @classmethod
    def value_memory_usage_MB(cls, value: Any): # -> str | None:
        """ returns the formatted memory size in MB. An example `value_formatting_fn` """
        ...
    


def generate_html_string(input_str, color=..., font_size=..., bold=..., italic=...): # -> str:
    """Generate an HTML string for use in a pyqtgraph label or title from an input string with optional formatting options.
    
    Args:
        input_str (str): The input string.
        color (str, optional): The color of the text. Defaults to None.
        font_size (str, optional): The font size of the text. Defaults to None.
        bold (bool, optional): Whether the text should be bold. Defaults to False.
        italic (bool, optional): Whether the text should be italic. Defaults to False.
    
    Returns:
        str: The HTML string.

    Usage:
        from pyphocorehelpers.print_helpers import generate_html_string
        i_str = generate_html_string('i', color='white', bold=True)
        j_str = generate_html_string('j', color='red', bold=True)
        title_str = generate_html_string(f'JSD(p_x_given_n, pf[{i_str}]) - JSD(p_x_given_n, pf[{j_str}]) where {j_str} non-firing')
        win.setTitle(title_str)

        >> 'JSD(p_x_given_n, pf[<b><span style="color:white;">i</span></b>]) - JSD(p_x_given_n, pf[<b><span style="color:red;">j</span></b>]) where <b><span style="color:red;">j</span></b> non-firing'
    """
    ...

def get_now_day_str() -> str:
    ...

def get_now_time_str(time_separator=...) -> str:
    ...

def get_now_time_precise_str(time_separator=...) -> str:
    ...

def get_now_rounded_time_str(rounded_minutes: float = ..., time_separator=...) -> str:
    """ rounded_minutes:float=2.5 - nearest previous minute mark to round to
    """
    ...

def split_seconds_human_readable(seconds): # -> tuple[int, int, int, Any | None]:
    """ splits the seconds argument into hour, minute, seconds, and fractional_seconds components.
        Does no formatting itself, but used by format_seconds_human_readable(...) for formatting seconds as a human-redable HH::MM:SS.FRACTIONAL time. 
    """
    ...

def format_seconds_human_readable(seconds, h_m_s_format_array=..., fixed_width=...): # -> tuple[int, int, int, Any | None, str | LiteralString]:
    """ returns the formatted string built from the seconds argument as a human-redable HH::MM:SS.FRACTIONAL time. 
    
    fixed_width: bool - if True, always returns HH:MM:SS.sss components even if the hours, minutes, etc are zero. Otherwise it returns starting with the MSB non-zero component
 
    Usage:
        test_seconds_array = [0, 10, 95, 1503, 543812]

        for test_seconds in test_seconds_array:
            print_seconds_human_readable(test_seconds, fixed_width=True)
            print_seconds_human_readable(test_seconds)

        >> Output >>
            00:00:00
            00
            00:00:10
            10
            00:01:35
            01:35
            00:25:03
            25:03
            151:03:32
            151:03:32

     """
    ...

def print_seconds_human_readable(seconds, h_m_s_format_array=..., fixed_width=...): # -> tuple[int, int, int, Any | None, str | LiteralString]:
    """ prints the seconds arguments as a human-redable HH::MM:SS.FRACTIONAL time. """
    ...

def print_dataframe_memory_usage(df, enable_print=...):
    """ df: a Pandas.DataFrame such as curr_active_pipeline.sess.spikes_df
    
    Usage:
        from pyphocorehelpers.print_helpers import print_dataframe_memory_usage
        print_dataframe_memory_usage(curr_active_pipeline.sess.spikes_df)

    >> prints >>:        
        ======== print_dataframe_memory_usage(df): ========
        Index                 0.00 MB
        t                     7.12 MB
        t_seconds             7.12 MB
        t_rel_seconds         7.12 MB
        shank                 3.56 MB
        cluster               3.56 MB
        aclu                  3.56 MB
        qclu                  3.56 MB
        x                     7.12 MB
        y                     7.12 MB
        speed                 7.12 MB
        traj                  3.56 MB
        lap                   3.56 MB
        maze_relative_lap     3.56 MB
        maze_id               3.56 MB
        neuron_type            35.58 MB
        flat_spike_idx        3.56 MB
        x_loaded              7.12 MB
        y_loaded              7.12 MB
        lin_pos               7.12 MB
        fragile_linear_neuron_IDX               3.56 MB
        PBE_id                7.12 MB
        dtype: object
        ============================
        Dataframe Total: 142.303 MB
    """
    ...

def print_object_memory_usage(obj, enable_print=...): # -> float:
    """ prints the size of the passed in object in MB (Megabytes)
    Usage:
        print_object_memory_usage(curr_bapun_pipeline.sess)
    """
    ...

def print_filesystem_file_size(file_path, enable_print=...): # -> float:
    """ prints the size of the file represented by the passed in path (if it exists) in MB (Megabytes)
    Usage:
        from pyphocorehelpers.print_helpers import print_filesystem_file_size
        print_filesystem_file_size(global_computation_results_pickle_path)
    """
    ...

def debug_print(*args, **kwargs): # -> None:
    ...

def print_callexp(*args, **kwargs): # -> None:
    """ DOES NOT WORK FROM Jupyter-lab notebook, untested in general.
    https://stackoverflow.com/questions/28244921/how-can-i-get-the-calling-expression-of-a-function-in-python?noredirect=1&lq=1
    
    """
    ...

def dbg_dump(*args, dumpopt_stream=..., dumpopt_forcename=..., dumpopt_pformat=..., dumpopt_srcinfo=..., **kwargs):
    """ DOES NOT WORK FROM Jupyter-lab notebook, untested in general.
    # pydump
    # A Python3 pretty-printer that also does introspection to detect the original
    # name of the passed variables
    #
    # Jean-Charles Lefebvre <polyvertex@gmail.com>
    # Latest version at: http://gist.github.com/polyvertex (pydump)

    Pretty-format every passed positional and named parameters, in that order,
    prefixed by their **original** name (i.e.: the one used by the caller), or
    by their type name for literals.
    Depends on the ``pprint``, ``inspect`` and ``ast`` standard modules.
    Note that the names of the keyword arguments you want to dump must not begin
    with ``dumpopt_`` since this prefix is used internally to differentiate
    options over values to dump.
    Also, the introspection code won't behave as expected if you make recursive
    calls to this function.
    Options can be passed as keyword arguments to tweak behavior and output
    format:
    * ``dumpopt_stream``:
      May you wish to print() the result directly, you can pass a stream object
      (e.g.: ``sys.stdout``) through this option, that will be given to
      ``print()``'s ``file`` keyword argument.
      You can also specify None in case you just want the output string to be
      returned without further ado.
    * ``dumpopt_forcename``:
      A boolean value to indicate wether you want every dumped value to be
      prepended by its name (i.e.: its name or its type).
      If ``False``, only non-literal values will be named.
    * ``dumpopt_pformat``:
      The dictionary of keyword arguments to pass to ``pprint.pformat()``
    * ``dumpopt_srcinfo``:
      Specify a false value (``None``, ``False``, zero) to skip caller's info.
      Specify ``1`` to output caller's line number only.
      Specify ``2`` to output caller's file name and line number.
      Specify ``3`` or greater to output caller's file path and line number.
    Example:
        ``dbg_dump(my_var, None, True, 123, "Bar", (4, 5, 6), fcall(), hello="world")``
    Result:
    ::
    DUMP(202):
        my_var: 'Foo'
        None: None
        Bool: True
        Num: 123
        Str: 'Bar'
        Tuple: (4, 5, 6)
        fcall(): "Function's Result"
        hello: 'world'
    """
    ...

class TypePrintMode(ExtendedEnum):
    """Describes the various ways of formatting an objects  type identity (`type(obj)`)
    Used by `print_file_progress_message(...)`
    """
    FULL_TYPE_STRING = ...
    FULL_TYPE_FQDN = ...
    TYPE_NAME_ONLY = ...
    def convert(self, curr_str: str, new_type) -> str:
        """ Converts from a more complete TypePrintMode down to a less complete one 

        Testing:
            TypePrintMode.FULL_TYPE_STRING.convert("<class 'pandas.core.frame.DataFrame'>", new_type=TypePrintMode.FULL_TYPE_FQDN) == 'pandas.core.frame.DataFrame'
            TypePrintMode.FULL_TYPE_STRING.convert("<class 'pandas.core.frame.DataFrame'>", new_type=TypePrintMode.FULL_TYPE_STRING) == "<class 'pandas.core.frame.DataFrame'>" # unaltered
            TypePrintMode.FULL_TYPE_STRING.convert("<class 'pandas.core.frame.DataFrame'>", new_type=TypePrintMode.TYPE_NAME_ONLY) == 'DataFrame'

        """
        ...
    


def strip_type_str_to_classname(a_type_str: str) -> str:
    """ Extracts the class string out of the string returned by type(an_obj) 
    a_type_str: a string returned by type(an_obj) in the form of ["<class 'tuple'>", "<class 'int'>", "<class 'float'>", "<class 'numpy.ndarray'>", "<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>", "<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>", "<class 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters'>"]
    return: str
    
    Example:
        test_input_class_strings = ["<class 'tuple'>", "<class 'int'>", "<class 'float'>", "<class 'numpy.ndarray'>", "<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>", "<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>", "<class 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters'>"]
        m = [strip_type_str_to_classname(a_test_str) for a_test_str in test_input_class_strings]
        print(m)        
        >> ['tuple', 'int', 'float', 'numpy.ndarray', 'pandas.core.series.Series', 'pandas.core.frame.DataFrame', 'pyphocorehelpers.indexing_helpers.BinningInfo', 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters']

    """
    ...

def safe_get_variable_shape(a_value): # -> _Shape | tuple[()] | int | None:
    """ generally and safely tries several methods of determining a_value's shape 
    
    assert safe_get_variable_shape(active_one_step_decoder.time_bin_size) is None
    assert isinstance(safe_get_variable_shape(active_one_step_decoder.spikes_df), tuple)
    assert isinstance(safe_get_variable_shape(active_one_step_decoder.F), tuple)
    """
    ...

_GLOBAL_DO_NOT_EXPAND_CLASS_TYPES = ...
_GLOBAL_DO_NOT_EXPAND_CLASSNAMES = ...
_GLOBAL_MAX_DEPTH = ...
def print_keys_if_possible(curr_key, curr_value, max_depth=..., depth=..., omit_curr_item_print=..., additional_excluded_item_classes=..., non_expanded_item_keys=..., custom_item_formatter=...): # -> None:
    """Prints the keys of an object if possible, in a recurrsive manner.

    Args:
        curr_key (str): the current key
        curr_value (_type_): the current value
        depth (int, optional): _description_. Defaults to 0.
        additional_excluded_item_classes (list, optional): A list of class types to exclude
        non_expanded_item_keys (list, optional): a list of keys which will not be expanded, no matter their type, only themselves printed.
        custom_item_formater (((depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False) -> str), optional): e.g. , custom_item_formatter=(lambda depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False: f"{depth_string}- {curr_key}: {type_name}")

            custom_item_formater Examples:
                from pyphocorehelpers.print_helpers import TypePrintMode
                print_keys_if_possible('computation_config', curr_active_pipeline.computation_results['maze1'].computation_config, custom_item_formatter=(lambda depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False: f"{depth_string}- {curr_key}: <{TypePrintMode.FULL_TYPE_STRING.convert(type_string, new_type=TypePrintMode.TYPE_NAME_ONLY)}>{' (children omitted)' if is_omitted_from_expansion else ''}))
                ! See `DocumentationFilePrinter._plain_text_format_curr_value` and `DocumentationFilePrinter._rich_text_format_curr_value` for further examples 

    Returns:
        None
        
    Usage:
        print_keys_if_possible('computed_data', curr_computations_results.computed_data, depth=0)
        
        - computed_data: <class 'dict'>
            - pf1D: <class 'neuropy.analyses.placefields.PfND'>
            - pf2D: <class 'neuropy.analyses.placefields.PfND'>
            - pf2D_Decoder: <class 'pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BayesianPlacemapPositionDecoder'>
            - pf2D_TwoStepDecoder: <class 'dict'>
                - xbin: <class 'numpy.ndarray'> - (59,)
                - ybin: <class 'numpy.ndarray'> - (21,)
                - avg_speed_per_pos: <class 'numpy.ndarray'> - (59, 21)
                - K: <class 'numpy.float64'>
                - V: <class 'float'>
                - sigma_t_all: <class 'numpy.ndarray'> - (59, 21)
                - all_x: <class 'numpy.ndarray'> - (59, 21, 2)
                - flat_all_x: <class 'list'>
                - original_all_x_shape: <class 'tuple'>
                - flat_p_x_given_n_and_x_prev: <class 'numpy.ndarray'> - (1239, 1717)
                - p_x_given_n_and_x_prev: <class 'numpy.ndarray'> - (59, 21, 1717)
                - most_likely_positions: <class 'numpy.ndarray'> - (2, 1717)
                - all_scaling_factors_k: <class 'numpy.ndarray'> - (1717,)
            - extended_stats: <class 'dict'>
                - time_binned_positioned_resampler: <class 'pandas.core.resample.TimedeltaIndexResampler'>
                - time_binned_position_df: <class 'pandas.core.frame.DataFrame'> - (1717, 18)
                - time_binned_position_mean: <class 'pandas.core.frame.DataFrame'> - (29, 16)
                - time_binned_position_covariance: <class 'pandas.core.frame.DataFrame'> - (16, 16)
            - firing_rate_trends: <class 'dict'>
                - active_rolling_window_times: <class 'pandas.core.indexes.timedeltas.TimedeltaIndex'>
                - mean_firing_rates: <class 'numpy.ndarray'> - (39,)
                - desired_window_length_seconds: <class 'float'>
                - desired_window_length_bins: <class 'int'>
                - active_firing_rates_df: <class 'pandas.core.frame.DataFrame'> - (1239, 39)
                - moving_mean_firing_rates_df: <class 'pandas.core.frame.DataFrame'> - (1239, 39)
            - placefield_overlap: <class 'dict'>
                - all_pairwise_neuron_IDs_combinations: <class 'numpy.ndarray'> - (741, 2)
                - total_pairwise_overlaps: <class 'numpy.ndarray'> - (741,)
                - all_pairwise_overlaps: <class 'numpy.ndarray'> - (741, 59, 21)
        
        ## Defining custom formatting functions:
            def _format_curr_value(depth_string, curr_key, curr_value, type_string, type_name):
                return f"{depth_string}['{curr_key}']: {type_name}"                
        
            print_keys_if_possible('active_firing_rate_trends', active_firing_rate_trends, custom_item_formatter=_format_curr_value)
        
            ['active_firing_rate_trends']: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
                ['time_bin_size_seconds']: float
                ['all_session_spikes']: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
                    ['time_window_edges']: numpy.ndarray - (5784,)
                    ['time_window_edges_binning_info']: pyphocorehelpers.indexing_helpers.BinningInfo
                        ['variable_extents']: tuple - (2,)
                        ['step']: float
                        ['num_bins']: int
                        ['bin_indicies']: numpy.ndarray - (5784,)
                    ['time_binned_unit_specific_binned_spike_rate']: pandas.core.frame.DataFrame - (5783, 52)
                    ['min_spike_rates']: pandas.core.series.Series - (52,)
                    ['median_spike_rates']: pandas.core.series.Series - (52,)
                    ['max_spike_rates']: pandas.core.series.Series - (52,)
                ['pf_included_spikes_only']: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
                    ['time_window_edges']: numpy.ndarray - (5779,)
                    ['time_window_edges_binning_info']: pyphocorehelpers.indexing_helpers.BinningInfo
                        ['variable_extents']: tuple - (2,)
                        ['step']: float
                        ['num_bins']: int
                        ['bin_indicies']: numpy.ndarray - (5779,)
                    ['time_binned_unit_specific_binned_spike_rate']: pandas.core.frame.DataFrame - (5778, 52)
                    ['min_spike_rates']: pandas.core.series.Series - (52,)
                    ['median_spike_rates']: pandas.core.series.Series - (52,)
                    ['max_spike_rates']: pandas.core.series.Series - (52,)

    
    """
    ...

def debug_dump_object_member_shapes(obj): # -> None:
    """ prints the name, type, and shape of all member variables. 
    Usage:
        debug_dump_object_member_shapes(active_one_step_decoder)
        >>>
            time_bin_size:	||	SCALAR	||	<class 'float'>
            pf:	||	SCALAR	||	<class 'neuropy.analyses.placefields.PfND'>
            spikes_df:	||	np.shape: (819170, 21)	||	<class 'pandas.core.frame.DataFrame'>
            debug_print:	||	SCALAR	||	<class 'bool'>
            neuron_IDXs:	||	np.shape: (64,)	||	<class 'numpy.ndarray'>
            neuron_IDs:	||	np.shape: (64,)	||	<class 'list'>
            F:	||	np.shape: (1856, 64)	||	<class 'numpy.ndarray'>
            P_x:	||	np.shape: (1856, 1)	||	<class 'numpy.ndarray'>
            unit_specific_time_binned_spike_counts:	||	np.shape: (64, 1717)	||	<class 'numpy.ndarray'>
            time_window_edges:	||	np.shape: (1718,)	||	<class 'numpy.ndarray'>
            time_window_edges_binning_info:	||	SCALAR	||	<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>
            total_spike_counts_per_window:	||	np.shape: (1717,)	||	<class 'numpy.ndarray'>
            time_window_centers:	||	np.shape: (1717,)	||	<class 'numpy.ndarray'>
            time_window_center_binning_info:	||	SCALAR	||	<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>
            flat_p_x_given_n:	||	np.shape: (1856, 1717)	||	<class 'numpy.ndarray'>
            p_x_given_n:	||	np.shape: (64, 29, 1717)	||	<class 'numpy.ndarray'>
            most_likely_position_flat_indicies:	||	np.shape: (1717,)	||	<class 'numpy.ndarray'>
            most_likely_position_indicies:	||	np.shape: (2, 1717)	||	<class 'numpy.ndarray'>
        <<< (end output example)
    """
    ...

def print_value_overview_only(a_value, should_return_string=...): # -> str | None:
    """ prints only basic information about a value, such as its type and shape if it has one. 
    
    Usage:
    
        test_value_1 = np.arange(15)
        print_value_overview_only(test_value_1)

        test_value_1 = list(range(15))
        print_value_overview_only(test_value_1)

        test_value_1 = 15
        print_value_overview_only(test_value_1)
            
        test_value_1 = 'test_string'
        print_value_overview_only(test_value_1)

        test_value_1 = {'key1': 0.34, 'key2': 'a'}
        print_value_overview_only(test_value_1)


    Note:
        str(value_type) => "<class 'numpy.ndarray'>"
        value_type.__name__ => 'ndarray'
        str(test_value_1.__class__).split("'") => ['<class ', 'numpy.ndarray', '>']
    """
    ...

def min_mean_max_sum(M, print_result=...): # -> tuple[Any, Any, Any, Any]:
    """Computes the min, mean, max, and sum of a matrix M (ignoring NaN values) and returns a tuple containing the results. Optionally can print the values.
    Useful for getting a simple summary/overview of a matrix.

    Args:
        M (_type_): _description_
        print_result (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    ...

def document_active_variables(params, include_explicit_values=..., enable_print=...): # -> dict[Any, Any]:
    """ Builds a skeleton for documenting variables and parameters by using the values set for a passed in instance.
    
    TODO: UNFINISHED!! UNTESTED.
    
    Usage:
        document_active_variables(active_curve_plotter_3d.params, enable_print=True)
    """
    ...

def get_system_hostname(enable_print: bool = ...) -> str:
    ...

def build_run_log_task_identifier(run_context: Union[str, List[str]], logging_root_FQDN: str = ..., include_curr_time_str: bool = ..., include_hostname: bool = ..., additional_suffix: Optional[str] = ...) -> str:
    """ Builds an identifier string for logging task progress like 'LNX00052.kdiba.gor01.two.2006-6-07_16-40-19'
    

    Usage:    
        from pyphocorehelpers.print_helpers import build_run_log_task_identifier

        build_run_log_task_identifier('test')
        
        >>> '2024-05-01_14-05-31.Apogee.com.PhoHale.Spike3D.test'

        build_run_log_task_identifier('test', logging_root_FQDN='Spike3D') # '2024-05-01_14-05-26.Apogee.Spike3D.test'
    
    """
    ...

def build_logger(full_logger_string: str, file_logging_dir=..., logFormatter: Optional[logging.Formatter] = ..., debug_print=...): # -> Logger:
    """ builds a logger
    
    from pyphocorehelpers.print_helpers import build_run_log_task_identifier, build_logger

    Default used to be:
        file_logging_dir=Path('EXTERNAL/TESTING/Logging')
    """
    ...

def ellided_dataframe(df: pd.DataFrame, max_rows_to_include: int = ..., num_truncated_rows_per_ellipsis_rows: int = ...) -> pd.DataFrame:
    """ returns a truncated/elided dataframe if the number of rows exceeds `max_rows_to_include`. Returned unchanged if n_rows is already less than `max_rows_to_include`.
    
    from pyphocorehelpers.print_helpers import ellided_dataframe
    
    truncated_df = ellided_dataframe(df=df, max_rows_to_include=100)
    """
    ...

def estimate_rendered_df_table_height(df: pd.DataFrame, debug_print=...) -> float:
    """ estimates the required height in px for rendered df
    
    Usage:
        estimated_table_height = estimate_rendered_df_table_height(df=normalized_df)
        will_scroll_vertically: bool = (estimated_table_height >= max_height)
    
    """
    ...

def render_scrollable_colored_table_from_dataframe(df: pd.DataFrame, cmap_name: str = ..., max_height: Optional[int] = ..., width: str = ..., is_dark_mode: bool = ..., max_rows_to_render_for_performance: int = ..., output_fn=..., **kwargs) -> Union[HTML, str]:
    """ Takes a numpy array of values and returns a scrollable and color-coded table rendition of it

    Usage:    
        from pyphocorehelpers.print_helpers import render_scrollable_colored_table_from_dataframe

        # Example usage:

        # Example 2D NumPy array
        array = np.random.rand(100, 10)
        # Draw it
        render_scrollable_colored_table(array)
        
        # Example 2:
            render_scrollable_colored_table(np.random.rand(100, 10), cmap_name='plasma', max_height=500, width='80%')
            render_scrollable_colored_table_from_dataframe(df=normalized_df, cmap_name=cmap_name, max_height=max_height, width=width, **kwargs)
            
    """
    ...

def render_scrollable_colored_table(array: NDArray, cmap_name: str = ..., max_height: Optional[int] = ..., width: str = ..., **kwargs) -> Union[HTML, str]:
    """ Takes a numpy array of values and returns a scrollable and color-coded table rendition of it

    Usage:    
        from pyphocorehelpers.print_helpers import render_scrollable_colored_table

        # Example 2D NumPy array
        array = np.random.rand(100, 10)
        # Draw it
        render_scrollable_colored_table(array)

    """
    ...

def array_preview_with_shape(arr): # -> None:
    """ Text-only Represntation that prints np.shape(arr) 
    
        from pyphocorehelpers.print_helpers import array_preview_with_shape

        # Register the custom display function for numpy arrays
        import IPython
        ip = IPython.get_ipython()
        ip.display_formatter.formatters['text/html'].for_type(np.ndarray, array_preview_with_shape) # only registers for NDArray

        # Example usage
        arr = np.random.rand(3, 4)
        display(arr)

    """
    ...

def array_preview_with_graphical_shape_repr_html(arr): # -> DisplayHandle | None:
    """Generate an HTML representation for a NumPy array, similar to Dask.
        
    from pyphocorehelpers.print_helpers import array_preview_with_graphical_shape_repr_html
    
    # Register the custom display function for NumPy arrays
    import IPython
    ip = IPython.get_ipython()
    ip.display_formatter.formatters['text/html'].for_type(np.ndarray, lambda arr: array_preview_with_graphical_shape_repr_html(arr))

    # Example usage
    arr = np.random.rand(3, 4)
    display(arr)


    arr = np.random.rand(9, 64)
    display(arr)

    arr = np.random.rand(9, 64, 4)
    display(arr)

    """
    ...

def array_preview_with_heatmap_repr_html(arr, include_shape: bool = ..., horizontal_layout=..., include_plaintext_repr: bool = ..., **kwargs): # -> str:
    """ Generate an HTML representation for a NumPy array with a Dask shape preview and a thumbnail heatmap
    
        from pyphocorehelpers.print_helpers import array_preview_with_heatmap_repr_html

        # Register the custom display function for numpy arrays
        import IPython
        ip = IPython.get_ipython()
        ip.display_formatter.formatters['text/html'].for_type(np.ndarray, array_preview_with_heatmap) # only registers for NDArray

        # Example usage
        arr = np.random.rand(3, 4)
        display(arr)

    """
    ...

