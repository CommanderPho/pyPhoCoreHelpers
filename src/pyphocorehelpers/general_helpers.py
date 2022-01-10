from typing import List, Optional, OrderedDict  # for OrderedMeta
from enum import Enum



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

   
