import vedo
from vedo import Mesh, Cone, Plotter, printc, Glyph
from vedo import Rectangle, Lines, Plane, Axes, merge, colorMap # for StaticVedo_3DRasterHelper
from vedo import Volume, ProgressBar, show, settings

# from pyphocorehelpers.plotting.vedo_qt_helpers import MainVedoPlottingWindow


class VedoHelpers:
    """docstring for VedoHelpers."""

    @classmethod
    def recurrsively_get_use_bounds(cls, assembly_obj):
        """
        # should_use_bounds: a bool value indicating whether the root assembly_obj and all of its actors/children should use bounds or not
        
        Usage:
            print(f'for assembly object with {len(active_window_only_axes.actors)} child actors: assembly UseBounds: {active_window_only_axes.GetUseBounds()} and actors UseBounds: {recurrsively_get_use_bounds(active_window_only_axes)}')
            recurrsively_get_use_bounds(active_window_only_axes) 

        
        """
        return [an_actor.GetUseBounds() for an_actor in assembly_obj.actors]

    @classmethod
    def recurrsively_apply_use_bounds(cls, assembly_obj, should_use_bounds):
        """
        # should_use_bounds: a bool value indicating whether the root assembly_obj and all of its actors/children should use bounds or not
        
        Usage:
            recurrsively_apply_use_bounds(active_window_only_axes, False)
        
        """
        assembly_obj.useBounds(should_use_bounds) # apply to parent object
        # originally_was_use_bounds = [an_actor.GetUseBounds() for an_actor in assembly_obj.actors]
        for an_actor in assembly_obj.actors:
            an_actor.useBounds(should_use_bounds)