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
