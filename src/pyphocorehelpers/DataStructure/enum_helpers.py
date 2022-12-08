from enum import Enum



class ExtendedEnum(Enum):
    """ Allows Inheritors to list their members, values, and names as lists

    Attribution:
        By blueFast answered Feb 28, 2019 at 5:58
        https://stackoverflow.com/a/54919285/9732163


    Usage:

    === For an Enum that inherits ExtendedEnum defined below:
    ```python
        @unique
        class FileProgressAction(ExtendedEnum):
            LOADING = "Loading"
            SAVING = "Saving"
            GENERIC = "Generic"

            @classmethod
            def init(cls, name):
                if name.upper() == cls.LOADING.name.upper():
                    return cls.LOADING
                elif name.upper() == cls.SAVING.name.upper():
                    return cls.SAVING
                elif name.upper() == cls.GENERIC.name.upper():
                    return cls.GENERIC
                else:
                    return cls.GENERIC
                    # raise NotImplementedError
                
            @property
            def actionVerb(self):
                return FileProgressAction.actionVerbsList()[self.value]

            # Static properties
            @classmethod
            def actionVerbsList(cls):
                return cls.build_member_value_dict(['from','to',':'])

    ```

    >>> Output
        FileProgressAction.all_members() # [<FileProgressAction.LOADING: 'Loading'>, <FileProgressAction.SAVING: 'Saving'>, <FileProgressAction.GENERIC: 'Generic'>]
        FileProgressAction.all_member_names() # ['LOADING', 'SAVING', 'GENERIC']
        FileProgressAction.all_member_values() # ['Loading', 'Saving', 'Generic']
        FileProgressAction.build_member_value_dict(['from','to',':']) # {<FileProgressAction.LOADING: 'Loading'>: 'from', <FileProgressAction.SAVING: 'Saving'>: 'to', <FileProgressAction.GENERIC: 'Generic'>: ':'}

    """
    # @classmethod
    # def to_list(cls):
    #     return list(map(lambda c: c.value, cls))

    @classmethod
    def all_members(cls) -> list:
        return list(map(lambda c: c, cls))

    @classmethod
    def all_member_names(cls) -> list:
        return list(map(lambda c: c.name, cls))

    @classmethod
    def all_member_values(cls) -> list:
        return list(map(lambda c: c.value, cls))

    @classmethod
    def build_member_value_dict(cls, values_list) -> dict:
        """ 
        
        values_list: <list>: e.g. ['from','to',':']

        """
        assert len(values_list) == len(cls.all_members()), f"there must be one value in values_list for each enum member but len(values_list): {len(values_list)} and len(cls.all_members): {len(cls.all_members())}."
        return dict(zip(cls.all_members(), values_list))


    # ==================================================================================================================== #
    # INIT Helpers                                                                                                         #
    # ==================================================================================================================== #
    @classmethod
    def _init_from_upper_name_dict(cls) -> dict:
        return dict(zip([a_name.upper() for a_name in cls.all_member_names()], cls.all_members()))
    @classmethod
    def _init_from_value_dict(cls) -> dict:
        return dict(zip(cls.all_member_values(), cls.all_members()))

    @classmethod
    def init(cls, name=None, value=None, fallback_value=None):
        """ Allows enum values to be initialized from either a name or value (but not both).

            e.g. FileProgressAction.init('lOaDing') # <FileProgressAction.LOADING: 'Loading'> 
        """
        assert (name is not None) or (value is not None), "You must specify either name or value, and the other will be returned"
        assert (name is None) or (value is None), "You cannot specify both name and value, as it would be ambiguous which takes priority. Please remove one of the two arguments."
        if name is not None:
            ## Name Mode:
            if fallback_value is not None:
                return cls._init_from_upper_name_dict().get(name.upper(), fallback_value)
            else:
                return cls._init_from_upper_name_dict()[name.upper()]
        elif value is not None:
            if fallback_value is not None:
                return cls._init_from_value_dict().get(value, fallback_value)
            else:
                return cls._init_from_value_dict()[value]
        else:
            raise NotImplementedError # THIS SHOULD NOT EVEN BE POSSIBLE!






class OrderedEnum(Enum):
    """ An enum that can be compared via comparison operators (like < and <=)
    Usage:
        class PipelineStage(OrderedEnum):
            Input = 0
            Loaded = 1
            Filtered = 2
            Computed = 3
            Displayed = 4
    
    """
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
