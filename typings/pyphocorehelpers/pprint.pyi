"""
This type stub file was generated by pyright.
"""

from contextlib import contextmanager
from io import StringIO
from IPython.utils.decorators import undoc

"""
Python advanced pretty printer.  This pretty printer is intended to
replace the old `pprint` python module which does not allow developers
to provide their own pretty print callbacks.

This module is based on ruby's `prettyprint.rb` library by `Tanaka Akira`.


Example Usage
-------------

To directly print the representation of an object use `pprint`::

    from pretty import pprint
    pprint(complex_object)

To get a string of the output use `pretty`::

    from pretty import pretty
    string = pretty(complex_object)


Extending
---------

The pretty library allows developers to add pretty printing rules for their
own objects.  This process is straightforward.  All you have to do is to
add a `_repr_pretty_` method to your object and call the methods on the
pretty printer passed::

    class MyObject(object):

        def _repr_pretty_(self, p, cycle):
            ...

Here's an example for a class with a simple constructor::

    class MySimpleObject:

        def __init__(self, a, b, *, c=None):
            self.a = a
            self.b = b
            self.c = c

        def _repr_pretty_(self, p, cycle):
            ctor = CallExpression.factory(self.__class__.__name__)
            if self.c is None:
                p.pretty(ctor(a, b))
            else:
                p.pretty(ctor(a, b, c=c))

Here is an example implementation of a `_repr_pretty_` method for a list
subclass::

    class MyList(list):

        def _repr_pretty_(self, p, cycle):
            if cycle:
                p.text('MyList(...)')
            else:
                with p.group(8, 'MyList([', '])'):
                    for idx, item in enumerate(self):
                        if idx:
                            p.text(',')
                            p.breakable()
                        p.pretty(item)

The `cycle` parameter is `True` if pretty detected a cycle.  You *have* to
react to that or the result is an infinite loop.  `p.text()` just adds
non breaking text to the output, `p.breakable()` either adds a whitespace
or breaks here.  If you pass it an argument it's used instead of the
default space.  `p.pretty` prettyprints another object using the pretty print
method.

The first parameter to the `group` function specifies the extra indentation
of the next line.  In this example the next item will either be on the same
line (if the items are short enough) or aligned with the right edge of the
opening bracket of `MyList`.

If you just want to indent something you can use the group function
without open / close parameters.  You can also use this code::

    with p.indent(2):
        ...

Inheritance diagram:

.. inheritance-diagram:: IPython.lib.pretty
   :parts: 3

:copyright: 2007 by Armin Ronacher.
            Portions (c) 2009 by Robert Kern.
:license: BSD License.
"""
__all__ = ['pretty', 'pprint', 'PrettyPrinter', 'RepresentationPrinter', 'for_type', 'for_type_by_name', 'RawText', 'RawStringLiteral', 'CallExpression']
MAX_LINE_LENGTH = ...
MAX_SEQ_LENGTH = ...
_re_pattern_type = ...
@undoc
class CUnicodeIO(StringIO):
    def __init__(self, *args, **kwargs) -> None:
        ...
    


def pretty(obj, verbose=..., max_width=..., newline=..., max_seq_length=...): # -> str:
    """
    Pretty print the object's representation.
    """
    ...

def pprint(obj, verbose=..., max_width=..., newline=..., max_seq_length=...): # -> None:
    """
    Like `pretty` but print to stdout.
    """
    ...

def pprint(object, stream=..., indent=..., width=..., depth=..., *, compact=..., sort_dicts=...): # -> None:
    """Pretty-print a Python object to a stream [default is sys.stdout]."""
    ...

def pformat(object, indent=..., width=..., depth=..., *, compact=..., sort_dicts=...):
    """Format a Python object into a pretty-printed representation."""
    ...

class _PrettyPrinterBase:
    @contextmanager
    def indent(self, indent): # -> Generator[None, Any, None]:
        """with statement support for indenting/dedenting."""
        ...
    
    @contextmanager
    def group(self, indent=..., open=..., close=...): # -> Generator[None, Any, None]:
        """like begin_group / end_group but for the with statement."""
        ...
    


class PrettyPrinter(_PrettyPrinterBase):
    """
    Baseclass for the `RepresentationPrinter` prettyprinter that is used to
    generate pretty reprs of objects.  Contrary to the `RepresentationPrinter`
    this printer knows nothing about the default pprinters or the `_repr_pretty_`
    callback method.
    """
    def __init__(self, output, max_width=..., newline=..., max_seq_length=..., compact=...) -> None:
        ...
    
    def text(self, obj): # -> None:
        """Add literal text to the output."""
        ...
    
    def breakable(self, sep=...): # -> None:
        """
        Add a breakable separator to the output.  This does not mean that it
        will automatically break here.  If no breaking on this position takes
        place the `sep` is inserted which default to one space.
        """
        ...
    
    def break_(self): # -> None:
        """
        Explicitly insert a newline into the output, maintaining correct indentation.
        """
        ...
    
    def begin_group(self, indent=..., open=...): # -> None:
        """
        Begin a group.
        The first parameter specifies the indentation for the next line (usually
        the width of the opening text), the second the opening text.  All
        parameters are optional.
        """
        ...
    
    def end_group(self, dedent=..., close=...): # -> None:
        """End a group. See `begin_group` for more details."""
        ...
    
    def flush(self): # -> None:
        """Flush data that is left in the buffer."""
        ...
    


class RepresentationPrinter(PrettyPrinter):
    """
    Special pretty printer that has a `pretty` method that calls the pretty
    printer for a python object.

    This class stores processing data on `self` so you must *never* use
    this class in a threaded environment.  Always lock it or reinstanciate
    it.

    Instances also have a verbose flag callbacks can access to control their
    output.  For example the default instance repr prints all attributes and
    methods that are not prefixed by an underscore if the printer is in
    verbose mode.
    """
    def __init__(self, output, verbose=..., max_width=..., newline=..., singleton_pprinters=..., type_pprinters=..., deferred_pprinters=..., max_seq_length=..., compact=...) -> None:
        ...
    
    def pretty(self, obj): # -> object | None:
        """Pretty print the given object."""
        ...
    


class Printable:
    def output(self, stream, output_width):
        ...
    


class Text(Printable):
    def __init__(self) -> None:
        ...
    
    def output(self, stream, output_width):
        ...
    
    def add(self, obj, width): # -> None:
        ...
    


class Breakable(Printable):
    def __init__(self, seq, width, pretty) -> None:
        ...
    
    def output(self, stream, output_width):
        ...
    


class Group(Printable):
    def __init__(self, depth) -> None:
        ...
    


class GroupQueue:
    def __init__(self, *groups) -> None:
        ...
    
    def enq(self, group): # -> None:
        ...
    
    def deq(self): # -> None:
        ...
    
    def remove(self, group): # -> None:
        ...
    


class RawText:
    """ Object such that ``p.pretty(RawText(value))`` is the same as ``p.text(value)``.

    An example usage of this would be to show a list as binary numbers, using
    ``p.pretty([RawText(bin(i)) for i in integers])``.
    """
    def __init__(self, value) -> None:
        ...
    


class CallExpression:
    """ Object which emits a line-wrapped call expression in the form `__name(*args, **kwargs)` """
    def __init__(__self, __name, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def factory(cls, name): # -> Callable[..., Self]:
        ...
    


class RawStringLiteral:
    """ Wrapper that shows a string with a `r` prefix """
    def __init__(self, value) -> None:
        ...
    


class _ReFlags:
    def __init__(self, value) -> None:
        ...
    


_exception_base = ...
_type_pprinters = ...
_env_type = ...
if _env_type is not dict:
    ...
_deferred_type_pprinters = ...
def for_type(typ, func): # -> None:
    """
    Add a pretty printer for a given type.
    """
    ...

def for_type_by_name(type_module, type_name, func): # -> None:
    """
    Add a pretty printer for a type specified by the module and name of a type
    rather than the type object itself.
    """
    ...

_singleton_pprinters = ...
def wide_pprint(*args, **kwargs): # -> None:
    """ it's mainly the `compact=True` that matters.
    From the documentation: 

        indent (default 1) specifies the amount of indentation added for each nesting level.

        depth controls the number of nesting levels which may be printed; if the data structure being printed is too deep, the next contained level is replaced by .... By default, there is no constraint on the depth of the objects being formatted.

        width (default 80) specifies the desired maximum number of characters per line in the output. If a structure cannot be formatted within the width constraint, a best effort will be made.

        compact impacts the way that long sequences (lists, tuples, sets, etc) are formatted. If compact is false (the default) then each item of a sequence will be formatted on a separate line. If compact is true, as many items as will fit within the width will be formatted on each output line.

        wide_pp = pprint.PrettyPrinter(width=_DESIRED_LINE_WIDTH, compact=True)
        
    """
    ...

def wide_pprint_ipython(obj, **kwargs): # -> None:
    ...

def wide_pprint_jupyter(obj, p, cycle, **kwargs): # -> None:
    ...

if __name__ == '__main__':
    class Foo:
        def __init__(self) -> None:
            ...
        
        def get_foo(self): # -> None:
            ...
        
    
    
