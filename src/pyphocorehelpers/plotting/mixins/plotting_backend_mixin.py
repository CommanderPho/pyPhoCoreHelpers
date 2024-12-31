# from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from enum import Enum


class PlottingBackendType(Enum):
    """Using Enums is a great way to make a dropdown menu."""
    PyQtGraph = "pyqtgraph"
    Matplotlib = "matplotlib"
    Other = "other"
    

# @metadata_attributes(short_name=None, tags=['plotting', 'backend', 'matplotlib', 'pyqtgraph', 'graphics'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 05:19', related_items=[])
class PlottingBackendSpecifyingMixin:
    """ 
    from pyphocorehelpers.plotting.mixins.plotting_backend_mixin import PlottingBackendSpecifyingMixin, PlottingBackendType
    
    """
    @classmethod
    def get_plot_backing_type(cls) -> PlottingBackendType:
        """PlottingBackendSpecifyingMixin conformance: Implementor should override and return either [PlottingBackendType.Matplotlib, PlottingBackendType.PyQtGraph]."""
        return PlottingBackendType.PyQtGraph
    
    @classmethod
    def is_pyqtgraph_based(cls) -> bool:
        return (cls.get_plot_backing_type().value == PlottingBackendType.PyQtGraph.value)
    
    @classmethod
    def is_matplotlib_based(cls) -> bool:
        return (cls.get_plot_backing_type().value == PlottingBackendType.Matplotlib.value)
    