from typing import List, Optional, OrderedDict  # for OrderedMeta
import numpy as np


class SimplePrintable:
    """Adds the default print method for classes that displays the class name and its dictionary."""
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__dict__};>"


class PrettyPrintable:
    def keys(self) -> List[Optional[str]]:
        return self.__dict__.keys()

    def _ipython_key_completions_(self) -> List[Optional[str]]:
        return self.keys()

    def _repr_pretty_(self, p, cycle=False):
        """The cycle parameter will be true if the representation recurses - e.g. if you put a container inside itself."""
        # p.text(self.__repr__() if not cycle else '...')
        p.text(self.__dict__.__repr__() if not cycle else "...")
        # return self.as_array().__repr__() # p.text(repr(self))


class WrappingMessagePrinter(object):
    """ 

    """
    def __init__(self, begin_string, begin_line_ending=' ', finished_message='done.', finished_line_ending='\n', returns_string:bool=False, enable_print:bool=True):
        self.begin_string = begin_string
        self.begin_line_ending = begin_line_ending
        self.finished_message = finished_message
        self.finished_line_ending = finished_line_ending
        
        self.returns_string = returns_string
        if self.returns_string:
            self.returned_string = ''
        else:
            self.returned_string = None    
        self.enable_print = enable_print
        
    def __enter__(self):
        self.returned_string = WrappingMessagePrinter.print_generic_progress_message(self.begin_string, self.begin_line_ending, self.returns_string, self.enable_print)
        # self.returned_string = WrappingMessagePrinter.print_file_progress_message(self.filepath, self.action, self.contents_description, self.print_line_ending, returns_string=self.returns_string)
        
    def __exit__(self, *args):
        if self.enable_print:
            print(self.finished_message, end=self.finished_line_ending)
        if self.returns_string:
            self.returned_string = f'{self.returned_string}{self.finished_message}{self.finished_line_ending}'
         
    @classmethod
    def print_generic_progress_message(cls, begin_string, begin_line_ending, returns_string, enable_print):
        if returns_string:
            out_string = f'{begin_string}...'
            if enable_print:
                print(out_string, end=begin_line_ending)
            return f'{out_string}{begin_line_ending}'
        else:
            if enable_print:
                print(f'{begin_string}...', end=begin_line_ending)
            
            
         
    # @classmethod   
    # def print_file_progress_message(cls, filepath, action: str, contents_description: str, print_line_ending=' ', returns_string=False):
    #     """[summary]
    #         print('Saving ripple epochs results to {}...'.format(ripple_epochs.filename), end=' ')
    #         ripple_epochs.save()
    #         print('done.')
            
    #     Args:
    #         filepath ([type]): [description]
    #         action (str): [description]
    #         contents_description (str): [description]
    #     """
    #     #  print_file_progress_message(ripple_epochs.filename, 'Saving', 'mua results') # replaces: print('Saving ripple epochs results to {}...'.format(ripple_epochs.filename), end=' ')
    #     if returns_string:
    #         out_string = f'{action} {contents_description} results to {str(filepath)}...'
    #         print(out_string, end=print_line_ending)
    #         return f'{out_string}{print_line_ending}'
    #     else:
    #         print(f'{action} {contents_description} results to {str(filepath)}...', end=print_line_ending)
        


def debug_dump_object_member_shapes(obj):
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
    for a_property_name, a_value in obj.__dict__.items():
        out_strings_arr = [f'{a_property_name}:']
        # np.isscalar(a_value)
        a_shape = np.shape(a_value)
        if a_shape != ():
            out_strings_arr.append(f'np.shape: {a_shape}')
        else:
            out_strings_arr.append(f'SCALAR')
            
        out_strings_arr.append(f'{str(type(a_value))}')
        out_string = '\t||\t'.join(out_strings_arr)
        print(out_string)


def print_seconds_human_readable(seconds):
    """ prints the seconds arguments as a human-redable HH::MM:SS.FRACTIONAL time. """
    if isinstance(seconds, int):
        whole_seconds = seconds
        fractional_seconds = None
    else:    
        whole_seconds = int(seconds)
        fractional_seconds = seconds - whole_seconds
    
    m, s = divmod(whole_seconds, 60)
    h, m = divmod(m, 60)
    timestamp = '{0:02}:{1:02}:{2:02}'.format(h, m, s)
    if fractional_seconds is not None:
        frac_seconds_string = ('%f' % fractional_seconds).rstrip('0').rstrip('.').lstrip('0').lstrip('.') # strips any insignficant zeros from the right, and then '0.' string from the left.        
        timestamp = '{}:{}'.format(timestamp, frac_seconds_string) # append the fracitonal seconds string to the timestamp string
    print(timestamp) # print the timestamp
    return h, m, s, fractional_seconds


def print_object_memory_usage(obj):
    """ prints the size of the passed in object in MB (Megabytes)
    Usage:
        print_object_memory_usage(curr_bapun_pipeline.sess)
    """
    size_bytes = obj.__sizeof__() # 1753723032
    size_MB = size_bytes/(1024*1024)
    object_size_string_MB = f'{size_MB} MB'
    print(f'object size: {object_size_string_MB}')
    return size_MB
