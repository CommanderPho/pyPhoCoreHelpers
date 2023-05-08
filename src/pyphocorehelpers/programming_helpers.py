# title: programming_helpers.py
# date: 2023-05-08 14:21:48
# purpose: Created to support programming and consolidation of programming-related helpers into a single location. Previously all were scattered around the various other helpers.


from functools import wraps
# from pyphocorehelpers.function_helpers import function_attributes, _custom_function_metadata_attribute_names

# ==================================================================================================================== #
# Documentation Decorators                                                                                             #
# ==================================================================================================================== #

def documentation_tags(func, *tags):
    """Adds documentations tags toa  function or class

    Usage:
        from pyphocorehelpers.programming_helpers import documentation_tags
        @documentation_tags('tag1', 'tag2')
        def my_function():
            ...

        Access via:
            my_function.tags

    """
    @wraps(func)
    def decorator(func):
        func.tags = tags
        return func
    return decorator


# ==================================================================================================================== #
# Function Attributes Decorators                                                                                       #
# ==================================================================================================================== #
_custom_metadata_attribute_names = dict(short_name=None, tags=None, creation_date=None,
                                         input_requires=None, output_provides=None,
                                         uses=None, used_by=None,
                                         related_items=None, # references to items related to this definition
                                         pyqt_signals_emitted=None # QtCore.pyqtSignal()s emitted by this object
)


def metadata_attributes(short_name=None, tags=None, creation_date=None, input_requires=None, output_provides=None, uses=None, used_by=None, related_items=None,  pyqt_signals_emitted=None):
    """Adds generic metadata attributes to a function or class
    Aims to generalize `pyphocorehelpers.function_helpers.function_attributes`

    ```python
        from pyphocorehelpers.programming_helpers import metadata_attributes

        @metadata_attributes(short_name='pf_dt_sequential_surprise', tags=['tag1','tag2'], input_requires=[], output_provides=[], uses=[], used_by=[])
        def _perform_time_dependent_pf_sequential_surprise_computation(computation_result, debug_print=False):
            # function body
    ```

    func.short_name, func.tags, func.creation_date, func.input_requires, func.output_provides, func.uses, func.used_by
    """
    # decorator = function_attributes(func) # get the decorator provided by function_attributes
    def decorator(func):
        func.short_name = short_name
        func.tags = tags
        func.creation_date = creation_date
        func.input_requires = input_requires
        func.output_provides = output_provides
        func.uses = uses
        func.used_by = used_by
        func.related_items = related_items
        func.pyqt_signals_emitted = pyqt_signals_emitted
        return func
    return decorator

