from enum import Enum

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
