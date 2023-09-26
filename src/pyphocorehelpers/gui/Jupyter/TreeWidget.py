from attrs import define, field, Factory
from collections import defaultdict

import ipytree as ipyt
from IPython.display import display


def _construct_hierarchical_dict_data(lst, keys):
    """ builds the hierarchical tree from the contexts:
    Usage:
        included_session_context_dict_tree = [ctxt.to_dict() for ctxt in included_session_contexts]
        keys = ['format_name', 'animal', 'exper_name', 'session_name']
        dict_list = included_session_context_dict_tree.copy()
        tree = _construct_hierarchical_dict_data(dict_list, keys)
        tree
    """
    tree = defaultdict(list)
    if len(keys) == 1:
        for d in lst:
            tree[d[keys[0]]].append(d)
        return dict(tree)
    for d in lst:
        tree[d[keys[0]]].append(d)
    for k, v in tree.items():
        tree[k] = _construct_hierarchical_dict_data(v, keys[1:])
    return dict(tree)



@define(slots=False)
class JupyterTreeWidget:
    """ displays a collapsable/expandable tree widget 

    Usage:
        from pyphocorehelpers.gui.Jupyter.TreeWidget import JupyterTreeWidget
        jupyter_tree_widget = JupyterTreeWidget(included_session_contexts=included_session_contexts)

    """
    included_session_contexts = field(default=Factory(list))
    tree = field(default=Factory(ipyt.Tree))
    max_depth = field(default=20)
    
    def __attrs_post_init__(self):
        self.construct_and_display_tree()


    def on_node_selected(self, change):
        if change['new']:
            selected_node = change['owner']
            # print(f"Selected node: {selected_node.name}")
            if hasattr(selected_node, 'context'):
                print(f"Selected context: {selected_node.context}")


    def build_tree(self, node, value, depth=0):
        """ constructs the ipytree nodes (GUI objects)"""
        if depth > self.max_depth:
            return

        if isinstance(value, dict):
            for key, child_value in value.items():
                child_node = ipyt.Node(key)
                child_node.selectable = False  # Make non-leaf nodes non-selectable
                node.add_node(child_node)
                self.build_tree(child_node, child_value, depth=depth+1)
        else:
            for context in value:
                leaf_node = ipyt.Node(str(context['session_name']))
                leaf_node.context = context  # Storing the original context as an attribute
                leaf_node.observe(self.on_node_selected, 'selected')
                node.add_node(leaf_node)


    def construct_and_display_tree(self):
        """ uses `self.included_session_contexts` to build the tree to display. """
        included_session_context_dict_tree = [ctxt.to_dict() for ctxt in self.included_session_contexts]
        assert len(included_session_context_dict_tree) > 0, f"tree cannot be empty but self.included_session_contexts: {self.included_session_contexts} "
        keys = list(included_session_context_dict_tree[0].keys())
        ## TODO: assert that they're equal for all entries?
        # keys = ['format_name', 'animal', 'exper_name', 'session_name']
        tree_data = _construct_hierarchical_dict_data(included_session_context_dict_tree, keys) # calls `_construct_hierarchical_dict_data`
        root_node = ipyt.Node('root')
        self.build_tree(root_node, tree_data)
        self.tree.add_node(root_node)
        display(self.tree)


# ==================================================================================================================== #
# Tree/Hierarchy Renderers and Previewers                                                                              #
# ==================================================================================================================== #


# def on_node_selected(change):
#     if change['new']:  # Node selected
#         print(f"Selected node: {change['owner'].name}")

# def build_tree(node, value, max_depth=20, depth=0):
#     """ 
#         import ipytree as ipyt
#         from IPython.display import display
#         from pyphocorehelpers.gui.Jupyter.TreeWidget import build_tree


#         # Assume curr_computations_results is the root of the data structure you want to explore
#         root_data = a_sess_config.preprocessing_parameters
#         root_node = ipyt.Node('sess.config')
#         build_tree(root_node, root_data)

#         tree = ipyt.Tree()
#         tree.add_node(root_node)
#         display(tree)
#     """
#     if depth > max_depth:
#         return

#     if isinstance(value, dict):
#         for key, child_value in value.items():
#             child_node = ipyt.Node(key)
#             node.add_node(child_node)
#             build_tree(child_node, child_value, max_depth=max_depth, depth=depth+1)
#     else:
#         ## add leaf node:
#         # node.add_node(ipyt.Node(str(value)))
#         leaf_node = ipyt.Node(str(value))
#         leaf_node.observe(on_node_selected, 'selected')
#         node.add_node(leaf_node)

