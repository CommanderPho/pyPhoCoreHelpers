from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable, iPythonKeyCompletingMixin
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters


class NestedRenderPlots(iPythonKeyCompletingMixin, DynamicParameters):
    """ 
        from pyphocorehelpers.DataStructure.NestedRenderPlots import NestedRenderPlots
    
    """
    
    @property
    def is_leaf(self):
        """ Whether this item is a leaf in the hierarchy or not """
        # return (len(self.render_items) > 0)
        return (not (len(self.children) > 0))
    
    @property
    def children(self):
        """The accessor for the TimeWindowPlaybackPropertiesMixin class for the main active time window that it will animate."""
        children_dict = {}
        for a_key, an_item in self.items():
            # Only return members that are NestedRenderPlots
            if isinstance(an_item, NestedRenderPlots):
                children_dict[a_key] = an_item
        return children_dict    

    def __init__(self, name, render_items=None, tags=None, **kwargs) -> None:
        if render_items is None:
            render_items = []
        if tags is None:
            tags = []
        super(NestedRenderPlots, self).__init__(name=name, render_items=render_items, tags=tags, **kwargs)

#     def __setattr__(self, attr, value):
#         if attr == '__setstate__':
#             return lambda: None
#         elif ((attr == '_mapping') or (attr == '_keys_at_init')):
#             # Access to the raw data variable
#             super(NestedRenderPlots, self).__setattr__(attr, value) # call super for valid properties
#         else:
#             if NestedRenderPlots.debug_enabled:
#                 print(f'NestedRenderPlots.__setattr__(self, attr, value): attr {attr}, value {value}')
                
#             # Wrap any child property that's attempted to be set in a version of itself first:
#             self[attr] = value

#     def __setitem__(self, key, value):
#         if NestedRenderPlots.debug_enabled:
#             print(f'NestedRenderPlots.__setitem__(self, key, value): key {key}, value {value}')
#         self._mapping[key] = value
        
    def get_hierarchy_render_items(self):
        """ Enumerates all the items in the render_items hierarchy as a flat list """
        curr_items = self.render_items # get this node's items
        curr_children_dict = self.children
        for a_child_key, a_child in curr_children_dict.items():
            curr_items = curr_items + a_child.get_hierarchy_render_items() # append child items
        return curr_items
    
    
    def get_hierarchy_render_item_paths(self, parent_path):
        """ Enumerates all the items in the render_items hierarchy as a flat list """
        # curr_path = parent_path.append(self.name)
        curr_path = parent_path.copy()
        curr_path.append(self.name)
        # print(f'curr_path: {curr_path}')
        
        if self.is_leaf:
            return [curr_path]
        else:
            curr_item_paths = [curr_path] # add this item, and then append its children
            curr_children_dict = self.children
            for a_child_key, a_child in curr_children_dict.items():
                curr_item_paths = curr_item_paths + a_child.get_hierarchy_render_item_paths(parent_path=curr_path) # append child items
            return curr_item_paths
    
    
    def add_item(self, obj, name='child', tags=None):
        """ adds the render item (obj) to this NestedRenderPlots's render_items """
        wrapped_obj = NestedRenderPlots.wrap_obj(obj, name=name, tags=tags)
        self.render_items.append(wrapped_obj) # adds the wrapped object to the render_items
        
        
    @classmethod
    def wrap_obj(cls, obj, name='child', tags=None):
        if isinstance(obj, NestedRenderPlots):
            return obj # don't double-wrap object
        else:
            # otherwise, single-wrap the object in a NestedRenderPlots
            return NestedRenderPlots(name, render_items=[obj], tags=tags)