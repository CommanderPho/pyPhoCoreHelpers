# ==================================================================================================================== #
# Tree/Hierarchy Renderers and Previewers                                                                              #
# ==================================================================================================================== #





import ipytree as ipyt
from IPython.display import display



def on_node_selected(change):
    if change['new']:  # Node selected
        print(f"Selected node: {change['owner'].name}")

def build_tree(node, value, max_depth=20, depth=0):
    """ 
        import ipytree as ipyt
        from IPython.display import display
        from pyphocorehelpers.gui.Jupyter.TreeWidget import build_tree


        # Assume curr_computations_results is the root of the data structure you want to explore
        root_data = a_sess_config.preprocessing_parameters
        root_node = ipyt.Node('sess.config')
        build_tree(root_node, root_data)

        tree = ipyt.Tree()
        tree.add_node(root_node)
        display(tree)
    """
    if depth > max_depth:
        return

    if isinstance(value, dict):
        for key, child_value in value.items():
            child_node = ipyt.Node(key)
            node.add_node(child_node)
            build_tree(child_node, child_value, max_depth=max_depth, depth=depth+1)
    else:
        ## add leaf node:
        # node.add_node(ipyt.Node(str(value)))
        leaf_node = ipyt.Node(str(value))
        leaf_node.observe(on_node_selected, 'selected')
        node.add_node(leaf_node)
