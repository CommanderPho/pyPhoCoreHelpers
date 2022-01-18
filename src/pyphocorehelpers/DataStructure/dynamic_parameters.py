
import collections


class DynamicParameters(collections.UserDict):
    """ A class that permits flexible prototyping of parameters and data needed for computations, while still allowing development-time guidance on available members.
    
        From https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/#When_making_a_custom_list_or_dictionary,_remember_you_have_options
        
        The UserDict class implements the interface that dictionaries are supposed to have, but it wraps around an actual dict object under-the-hood.

        The UserList and UserDict classes are for when you want something that acts almost identically to a list or a dictionary but you want to customize just a little bit of functionality.

        The abstract base classes in collections.abc are useful when you want something that’s a sequence or a mapping but is different enough from a list or a dictionary that you really should be making your own custom class.

    """

    def __init__(self, **kwargs):
        super(DynamicParameters, self).__init__()
        # Dump all arguments into parameters.
        for key, value in kwargs.items():
            setattr(self, key, value)

    # For collections.UserDict conformance:            
    def __delitem__(self, key):
        value = self.data.pop(key)
        self.data.pop(value, None)

    def __setitem__(self, key, value):
        if key in self:
            del self[self[key]]
        if value in self:
            del self[value]
        self.data[key] = value
        self.data[value] = key

    def __dir__(self):
        return self.keys()
 
    def __getattr__(self, attr):
        # Fake a __getstate__ method that returns None
        if attr == "__getstate__":
            return lambda: None
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value
        
    
    def __hash__(self):
        """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
        # return hash((self.age, self.name))
        member_names_tuple = list(self.__dict__.keys())
        values_tuple = list(self.__dict__.values())
        combined_tuple = tuple(member_names_tuple + values_tuple)
        return hash(combined_tuple)
    
    
    
    # def _unlisted_parameter_strings(self):
    #     """ returns the string representations of all key/value pairs that aren't normally defined. """
    #     # Dump all arguments into parameters.
    #     out_list = []
    #     for key, value in self.__dict__.items():
    #         if key not in PlacefieldComputationParameters.variable_names:
    #             out_list.append(f"{key}_{value:.2f}")
    #     return out_list