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




def get_arguments_as_optional_dict(**kwargs):
    """ Easily converts your existing argument-list style default values into a dict:
            Defines a simple function that takes only **kwargs as its inputs and prints the values it recieves. Paste your values as arguments to the function call. The dictionary will be output to the console, so you can easily copy and paste. 
        Usage:
            >>> get_arguments_as_optional_dict(point_size=8, font_size=10, name='build_center_labels_test', shape_opacity=0.8, show_points=False)

            Output: ", **({'point_size': 8, 'font_size': 10, 'name': 'build_center_labels_test', 'shape_opacity': 0.8, 'show_points': False} | kwargs)"
    """
    print(', **(' + f'{kwargs}' + ' | kwargs)')

