#TODO 2023-06-20 12:16: - [ ] Convert to `SubsettableDictRepresentable`
# from neuropy.utils.mixins.dict_representable import SubsettableDictRepresentable
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
    # Display Library Test Functions _____________________________________________________________________________________ #
    @classmethod
    def is_matplotlib(cls):
        """Whether the display library is matplotlib."""
        return cls._display_library == 'matplotlib'
    @classmethod
    def is_pyqtgraph(cls):
        """Whether the display library is pyqtgraph."""
        return cls._display_library == 'pyqtgraph'


class RenderPlotsData(iPythonKeyCompletingMixin, DynamicParameters):
    def __init__(self, name, **kwargs) -> None:
        super(RenderPlotsData, self).__init__(name=name, **kwargs)


