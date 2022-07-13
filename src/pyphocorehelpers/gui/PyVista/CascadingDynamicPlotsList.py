from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters


class CascadingDynamicPlotsList(DynamicParameters):
    """ 
    Creates a simple collection wrapper capable of having functions applied on its children.
    
    Currently used in pyphoplacecellanalysis.PhoPositionalData.plotting.spikeAndPositions.plot_placefields2D(...) to wrap multiple plot actors (for both the main pf mesh and the points)
    
    """
    def __init__(self, **kwargs):      
        super(CascadingDynamicPlotsList, self).__init__(**kwargs) # points=active_points_plotActor

    @property
    def plotActors(self):
        # Static list of plot actors
        # Exclude members like: '__class__'
        return [self[an_attr] for an_attr in self.all_attributes if not an_attr.startswith('_')]
    
    # def callFunctionOnChildren(self, functionToApply, *args, **kwargs):
    def callFunctionOnChildren(self, child_method_name, *args, **kwargs):
        """Calls the method with the name child_method_name for each child object and returns the results as a list.

        Args:
            child_method_name (str): a string containing the method to be called on each of the child objects

        Returns:
            _type_: _description_
        """
        curr_plot_actors = self.plotActors
        output_results = []
        for curr_plot_actor in curr_plot_actors:
            curr_output = getattr(curr_plot_actor, child_method_name)(*args, **kwargs)
            # curr_output = CascadingDynamicPlotsList.apply_on_all(curr_plot_actor, child_method_name, *args, **kwargs)
            # curr_output = functionToApply(curr_plot_actor, *args, **kwargs)
            output_results.append(curr_output)
        return output_results
    
    
    def SetVisibility(self, value):
        self.callFunctionOnChildren('SetVisibility', value)
        
    
    @staticmethod    
    def apply_on_all(seq, method, *args, **kwargs):
        """ Applies a method with the specified name to all objects in the sequence, passing the same *args and **kwargs
            by Ants Aasma, answered Apr 21, 2010 at 10:30 
        """
        for obj in seq:
            getattr(obj, method)(*args, **kwargs)
            

# ds