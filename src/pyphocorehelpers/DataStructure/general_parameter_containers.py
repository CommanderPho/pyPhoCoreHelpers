from pyphocorehelpers.print_helpers import iPythonKeyCompletingMixin
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

class VisualizationParameters(iPythonKeyCompletingMixin, DynamicParameters):
    def __init__(self, name, **kwargs) -> None:
        super(VisualizationParameters, self).__init__(name=name, **kwargs)
    

class DebugHelper(iPythonKeyCompletingMixin, DynamicParameters):
    def __init__(self, name, **kwargs) -> None:
        super(DebugHelper, self).__init__(name=name, **kwargs)


class RenderPlots(iPythonKeyCompletingMixin, DynamicParameters):
    def __init__(self, name, **kwargs) -> None:
        super(RenderPlots, self).__init__(name=name, **kwargs)


class RenderPlotsData(iPythonKeyCompletingMixin, DynamicParameters):
    def __init__(self, name, **kwargs) -> None:
        super(RenderPlotsData, self).__init__(name=name, **kwargs)


