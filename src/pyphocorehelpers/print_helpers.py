from typing import List, Optional, OrderedDict  # for OrderedMeta
import numpy as np

# Required for dbg_dump:
import sys
import pprint
import inspect
import ast

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
        

# def debug_print_shapes(*arg):
#     """ prints the shape of the passed arugments """


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


def debug_print(*args, **kwargs):
    # print(f'xbin_edges: {xbin_edges}\nxbin_centers: {xbin_centers}\nybin_edges: {ybin_edges}\nybin_centers: {ybin_centers}')
    out_strings = []
    for i, an_ordered_arg in enumerate(args):
        out_strings.append(f'args[{i}]: {args[i]}')
        
    for key, val in kwargs.items():
        out_strings.append(f'{key}: {val}')

    out_string = '\n'.join(out_strings)
    print(out_string)
    

def print_callexp(*args, **kwargs):
    """ DOES NOT WORK FROM Jupyter-lab notebook, untested in general.
    https://stackoverflow.com/questions/28244921/how-can-i-get-the-calling-expression-of-a-function-in-python?noredirect=1&lq=1
    
    """
    def _find_caller_node(root_node, func_name, last_lineno):
        # init search state
        found_node = None
        lineno = 0

        def _luke_astwalker(parent):
            nonlocal found_node
            nonlocal lineno
            for child in ast.iter_child_nodes(parent):
                # break if we passed the last line
                if hasattr(child, "lineno"):
                    lineno = child.lineno
                if lineno > last_lineno:
                    break

                # is it our candidate?
                if (isinstance(child, ast.Name)
                        and isinstance(parent, ast.Call)
                        and child.id == func_name):
                    # we have a candidate, but continue to walk the tree
                    # in case there's another one following. we can safely
                    # break here because the current node is a Name
                    found_node = parent
                    break

                # walk through children nodes, if any
                _luke_astwalker(child)

        # dig recursively to find caller's node
        _luke_astwalker(root_node)
        return found_node

    # get some info from 'inspect'
    frame = inspect.currentframe()
    backf = frame.f_back
    this_func_name = frame.f_code.co_name

    # get the source code of caller's module
    # note that we have to reload the entire module file since the
    # inspect.getsource() function doesn't work in some cases (i.e.: returned
    # source content was incomplete... Why?!).
    # --> is inspect.getsource broken???
    #     source = inspect.getsource(backf.f_code)
    #source = inspect.getsource(backf.f_code)
    with open(backf.f_code.co_filename, "r") as f:
        source = f.read()

    # get the ast node of caller's module
    # we don't need to use ast.increment_lineno() since we've loaded the whole
    # module
    ast_root = ast.parse(source, backf.f_code.co_filename)
    #ast.increment_lineno(ast_root, backf.f_code.co_firstlineno - 1)

    # find caller's ast node
    caller_node = _find_caller_node(ast_root, this_func_name, backf.f_lineno)

    # now, if caller's node has been found, we have the first line and the last
    # line of the caller's source
    if caller_node:
        #start_index = caller_node.lineno - backf.f_code.co_firstlineno
        #end_index = backf.f_lineno - backf.f_code.co_firstlineno + 1
        print("Hoooray! Found it!")
        start_index = caller_node.lineno - 1
        end_index = backf.f_lineno
        lineno = caller_node.lineno
        for ln in source.splitlines()[start_index:end_index]:
            print("  {:04d} {}".format(lineno, ln))
            lineno += 1

def dbg_dump(
        *args,
        dumpopt_stream=sys.stderr,
        dumpopt_forcename=True,
        dumpopt_pformat={'indent': 2},
        dumpopt_srcinfo=1,
        **kwargs):
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
    try:
        def _find_caller_node(root_node, func_name, last_lineno):
            # find caller's node by walking down the ast, searching for an
            # ast.Call object named func_name of which the last source line is
            # last_lineno
            found_node = None
            lineno = 0
            def _luke_astwalker(parent):
                nonlocal found_node
                nonlocal lineno
                for child in ast.iter_child_nodes(parent):
                    # break if we passed the last line
                    if hasattr(child, "lineno") and child.lineno:
                        lineno = child.lineno
                    if lineno > last_lineno:
                        break
                    # is it our candidate?
                    if (isinstance(child, ast.Name)
                            and isinstance(parent, ast.Call)
                            and child.id == func_name):
                        found_node = parent
                        break
                    _luke_astwalker(child)
            _luke_astwalker(root_node)
            return found_node

        frame = inspect.currentframe()
        backf = frame.f_back
        this_func_name = frame.f_code.co_name
        #this_func = backf.f_locals.get(
        #    this_func_name, backf.f_globals.get(this_func_name))

        # get the source code of caller's module
        # note that we have to reload the entire module file since the
        # inspect.getsource() function doesn't work in some cases (i.e.:
        # returned source content was incomplete... Why?!).
        # --> is inspect.getsource broken???
        #     source = inspect.getsource(backf.f_code)
        #source = inspect.getsource(backf.f_code)
        with open(backf.f_code.co_filename, "r") as f:
            source = f.read()

        # get the ast node of caller's module
        # we don't need to use ast.increment_lineno() since we've loaded the
        # whole module
        ast_root = ast.parse(source, backf.f_code.co_filename)
        #ast.increment_lineno(ast_root, backf.f_code.co_firstlineno - 1)

        # find caller's ast node
        caller_node = _find_caller_node(ast_root, this_func_name, backf.f_lineno)
        if not caller_node:
            raise Exception("caller's AST node not found")

        # keep some useful info for later
        src_info = {
            'file': backf.f_code.co_filename,
            'name': (
                backf.f_code.co_filename.replace("\\", "/").rpartition("/")[2]),
            'lineno': caller_node.lineno}

        # if caller's node has been found, we now have the AST of our parameters
        args_names = []
        for arg_node in caller_node.args:
            if isinstance(arg_node, ast.Name):
                args_names.append(arg_node.id)
            elif isinstance(arg_node, ast.Attribute):
                if hasattr(arg_node, "value") and hasattr(arg_node.value, "id"):
                    args_names.append(arg_node.value.id + "." + arg_node.attr)
                else:
                    args_names.append(arg_node.attr)
            elif isinstance(arg_node, ast.Subscript):
                args_names.append(arg_node.value.id + "[]")
            elif (isinstance(arg_node, ast.Call)
                    and hasattr(arg_node, "func")
                    and hasattr(arg_node.func, "id")):
                args_names.append(arg_node.func.id + "()")
            elif dumpopt_forcename:
                if (isinstance(arg_node, ast.NameConstant)
                        and arg_node.value is None):
                    args_names.append("None")
                elif (isinstance(arg_node, ast.NameConstant)
                        and arg_node.value in (False, True)):
                    args_names.append("Bool")
                else:
                    args_names.append(arg_node.__class__.__name__)
            else:
                args_names.append(None)
    except:
        #import traceback
        #traceback.print_exc()
        src_info = None
        args_names = [None] * len(args)

    args_count = len(args) + len(kwargs)

    output = ""
    if dumpopt_srcinfo:
        if not src_info:
            output += "DUMP(<unknown>):"
        else:
            if dumpopt_srcinfo <= 1:
                fmt = "DUMP({2}):"
            elif dumpopt_srcinfo == 2:
                fmt = "{1}({2}):"
            else:
                fmt = "{0}({2}):"
            output += fmt.format(
                        src_info['file'], src_info['name'], src_info['lineno'])
        output += "\n" if args_count > 1 else " "
    else:
        src_info = None

    for name, obj in zip(
            args_names + list(kwargs.keys()),
            list(args) + list(kwargs.values())):
        if name and name.startswith("dumpopt_"):
            continue
        if dumpopt_srcinfo and args_count > 1:
            output += "  "
        if name:
            output += name + ": "
        output += pprint.pformat(obj, **dumpopt_pformat) + "\n"

    if dumpopt_stream:
        print(output, end="", file=dumpopt_stream)
        return None # explicit is better than implicit
    else:
        return output.rstrip()