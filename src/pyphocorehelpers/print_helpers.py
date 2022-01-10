from typing import List, Optional, OrderedDict  # for OrderedMeta


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

