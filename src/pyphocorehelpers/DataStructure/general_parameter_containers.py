#TODO 2023-06-20 12:16: - [ ] Convert to `SubsettableDictRepresentable`
# from neuropy.utils.mixins.dict_representable import SubsettableDictRepresentable
from typing import List, Dict, Optional
from pyphocorehelpers.print_helpers import iPythonKeyCompletingMixin
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

class VisualizationParameters(iPythonKeyCompletingMixin, DynamicParameters):
    def __init__(self, name, **kwargs) -> None:
        super(VisualizationParameters, self).__init__(name=name, **kwargs)
    

class DebugHelper(iPythonKeyCompletingMixin, DynamicParameters):
    def __init__(self, name, **kwargs) -> None:
        super(DebugHelper, self).__init__(name=name, **kwargs)

#TODO 2023-06-13 10:59: - [ ] Convert to an attrs-based class instead of inheriting from DynamicParameters
# @attrs.define(slots=False)
class RenderPlots(iPythonKeyCompletingMixin, DynamicParameters):
    _display_library:str = 'unknown'
    def __init__(self, name, context=None, **kwargs) -> None:
        super(RenderPlots, self).__init__(name=name, context=context, **kwargs)

    @classmethod
    def non_data_keys(cls) -> List[str]:
        """ a list of the non-user-contributed keys. """
        return ['name', 'context'] # , '_display_library'

    def data_keys(self):
        return [k for k in self.keys() if k not in self.non_data_keys] 

    # Display Library Test Functions _____________________________________________________________________________________ #
    @classmethod
    def is_matplotlib(cls):
        """Whether the display library is matplotlib."""
        return cls._display_library == 'matplotlib'
    @classmethod
    def is_pyqtgraph(cls):
        """Whether the display library is pyqtgraph."""
        return cls._display_library == 'pyqtgraph'



    # def __add__(self, other):
    #     if not isinstance(other, CustomDict):
    #         raise TypeError("Unsupported operand type. The operand must be of type CustomDict.")
    #     combined_data = {key: self.data.get(key, 0) + other.data.get(key, 0) for key in set(self.data) | set(other.data)}
    #     return CustomDict(combined_data)


class RenderPlotsData(iPythonKeyCompletingMixin, DynamicParameters):
    def __init__(self, name, **kwargs) -> None:
        super(RenderPlotsData, self).__init__(name=name, **kwargs)


