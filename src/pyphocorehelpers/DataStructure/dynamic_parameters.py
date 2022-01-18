
import collections
from collections.abc import MutableMapping


class DynamicParameters(MutableMapping):
    """ A class that permits flexible prototyping of parameters and data needed for computations, while still allowing development-time guidance on available members.
    
        From https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/#When_making_a_custom_list_or_dictionary,_remember_you_have_options
        
        The UserDict class implements the interface that dictionaries are supposed to have, but it wraps around an actual dict object under-the-hood.

        The UserList and UserDict classes are for when you want something that acts almost identically to a list or a dictionary but you want to customize just a little bit of functionality.

        The abstract base classes in collections.abc are useful when you want something that’s a sequence or a mapping but is different enough from a list or a dictionary that you really should be making your own custom class.

    """
    debug_enabled = True

    def __init__(self, **kwargs):
        self.mapping = {} # initialize the base dictionary object where things will be stored
        self._keys_at_init = list(kwargs.keys())
        self.update(kwargs)
        # for key, value in kwargs.items():
        #     # setattr(self, key, value)
        #     self[key] = value

    def __getitem__(self, key):
        if DynamicParameters.debug_enabled:
            print(f'DynamicParameters.__getitem__(self, key): key {key}')
        return self.mapping[key]

    def __delitem__(self, key):
        if DynamicParameters.debug_enabled:
            print(f'DynamicParameters.__delitem__(self, key): key {key}')
        del self.mapping[key]

    def __setitem__(self, key, value):
        if DynamicParameters.debug_enabled:
            print(f'DynamicParameters.__setitem__(self, key, value): key {key}, value {value}')
        self.mapping[key] = value

    def __iter__(self):
        return iter(self.mapping)
    def __len__(self):
        return len(self.mapping)
    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"

    # Extra/Extended
    def __dir__(self):
        if DynamicParameters.debug_enabled:
            print(f'DynamicParameters.__dir__(self)')
        return self.keys()

    ## Enable access by object members:
#     def __getattr__(self, attr):
#         # Fake a __getstate__ method that returns None

#         # AttributeError: 'DynamicParameters' object has no attribute 'prop0'
#         if attr == "data":
#             # Access to the raw data variable
#             return self.data
#         else:
#             if DynamicParameters.debug_enabled:
#                 print(f'DynamicParameters.__getattr__(self, attr): attr {attr}')
#             return self[attr]


    # This works, but is un-needed
    # def __getattribute__(self, item):
    #      # Gets called when an attribute is accessed
    #     if DynamicParameters.debug_enabled:
    #         print(f'DynamicParameters.__getattribute__(self, item): item {item}')
    #     # Calling the super class to avoid recursion
    #     return super(DynamicParameters, self).__getattribute__(item)

    def __getattr__(self, item):
        # Gets called when the item is not found via __getattribute__
        if DynamicParameters.debug_enabled:
            print(f'DynamicParameters.__getattr__(self, item): item {item}')
        try:
            # try to return the value of the dictionary 
            return self[item]
        except AttributeError as err:
            print(f"AttributeError: {err}")
            return super(DynamicParameters, self).__setattr__(item, 'orphan')
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


#     def __setattr__(self, attr, value):
#         if attr == "__setstate__":
#             return lambda: None
#         elif attr == "data":
#             # Access to the raw data variable
#             self.data = value # to initialize the raw data
#         else:
#             if DynamicParameters.debug_enabled:
#                 print(f'DynamicParameters.__setattr__(self, attr, value): attr {attr}, value {value}')
#             self[attr] = value


#     def __hash__(self):
#         """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
#         # return hash((self.age, self.name))
#         member_names_tuple = list(self.__dict__.keys())
#         values_tuple = list(self.__dict__.values())
#         combined_tuple = tuple(member_names_tuple + values_tuple)
#         return hash(combined_tuple)
    
    
    # def _unlisted_parameter_strings(self):
    #     """ returns the string representations of all key/value pairs that aren't normally defined. """
    #     # Dump all arguments into parameters.
    #     out_list = []
    #     for key, value in self.__dict__.items():
    #         if key not in PlacefieldComputationParameters.variable_names:
    #             out_list.append(f"{key}_{value:.2f}")
    #     return out_list