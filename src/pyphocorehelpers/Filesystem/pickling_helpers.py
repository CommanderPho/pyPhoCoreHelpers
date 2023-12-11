import sys
import io
import types
# from types import ModuleType
from typing import List, Dict

# import pickle
# import dill
import dill as pickle
import pandas as pd

move_modules_list = {'pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper.SingleBarResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.SingleBarResult',
    'pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper.InstantaneousSpikeRateGroupsComputation':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.InstantaneousSpikeRateGroupsComputation',
    }


class RenameUnpickler(pickle.Unpickler):
    """ 
    # global_move_modules_list: Dict[str, str] - a dict with keys equal to the old full path to a class and values equal to the updated (replacement) full path to the class. Used to update the path to class definitions for loading previously pickled results after refactoring.
        Example: 	{'pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper.SingleBarResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.SingleBarResult',
                    'pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper.InstantaneousSpikeRateGroupsComputation':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.InstantaneousSpikeRateGroupsComputation',
                    }

    
        
    Usage:
        from pyphocorehelpers.Filesystem.pickling_helpers import RenameUnpickler, renamed_load
    

    pkl_path = curr_active_pipeline.global_computation_results_pickle_path
    with open(pkl_path, 'rb') as dbfile:
        # db = pickle.load(dbfile, **kwargs) # replace previous ` pickle.load(dbfile, **kwargs)` calls with `db = renamed_load(dbfile, **kwargs)`
        db = renamed_load(dbfile, **kwargs)
        
    """
    def find_class(self, module:str, name:str):
        original_full_name:str = '.'.join((module, name))
        renamed_module = module
        assert self._move_modules_list is not None
        
        found_full_replacement_name = self._move_modules_list.get(original_full_name, None)        
        if found_full_replacement_name is not None:
            found_full_replacement_module, found_full_replacement_import_name = found_full_replacement_name.rsplit('.', 1)
            renamed_module = found_full_replacement_module
            
        # Pandas 1.5.* -> 2.0.* pickle compatibility:
        # after upgrading Pandas from 1.5.* -> 2.0.* I was getting `ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'` when trying to unpickle the pipeline.
        if module.startswith('pandas.'):
            key = (module, name)
            renamed_module, name = self._pandas_rename_map.get(key, key)
    
        return super(RenameUnpickler, self).find_class(renamed_module, name)

    def __init__(self, *args, **kwds):
        # settings = pickle.Pickler.settings
        # _ignore = kwds.pop('ignore', None)
        _move_modules_list = kwds.pop('move_modules_list', None)        
        pickle.Unpickler.__init__(self, *args, **kwds)
        # self._ignore = settings['ignore'] if _ignore is None else _ignore
        # self._move_modules_list = settings['move_modules_list'] if _move_modules_list is None else _move_modules_list
        assert _move_modules_list is not None
        self._move_modules_list = _move_modules_list
        self._pandas_rename_map = pd.compat.pickle_compat._class_locations_map

def renamed_load(file_obj, move_modules_list:Dict=None, **kwargs):
    """ from pyphocorehelpers.Filesystem.pickling_helpers import renamed_load """
    return RenameUnpickler(file_obj, move_modules_list=move_modules_list, **kwargs).load()

def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)



# def update_module_path_in_pickled_object(pickle_path: str, old_module_path: str, new_module: ModuleType) -> None:
#     """Update a python module's dotted path in a pickle dump if the
#     corresponding file was renamed.

#     Implements the advice in https://stackoverflow.com/a/2121918.

#     Args:
#         pickle_path (str): Path to the pickled object.
#         old_module_path (str): The old.dotted.path.to.renamed.module.
#         new_module (ModuleType): from new.location import module.
#     """
#     sys.modules[old_module_path] = new_module

#     dic = pickle.load(open(pickle_path, "rb"))
#     # dic = torch.load(pickle_path, map_location="cpu")

#     del sys.modules[old_module_path]

#     pickle.dump(dic, open(pickle_path, "wb"))
#     # torch.save(dic, pickle_path)


exclude_modules_list = ['pyphoplacecellanalysis.External.pyqtgraph'] # , 'pyphoplacecellanalysis.External.pyqtgraph.Qt.QtWidgets', 'pyphoplacecellanalysis.External.pyqtgraph.Qt.QtWidgets.QApplication'
exclude_type_names = ["QApplication"]  # Add the type names you want to exclude as strings


class ModuleExcludesPickler(pickle.Pickler):
    """ 
    TypeError: cannot pickle 'QApplication' object
    
    """
    def save_module(self, obj, name=None):
        # Replace 'module_to_exclude' with the actual module name you want to exclude
        # valid_module_to_include: bool = True
        # for a_module_to_exclude in exclude_modules_list:
        #     if a_module_to_exclude in obj.__name__:
        #         valid_module_to_include = False
        #         return
        # if valid_module_to_include:
        #     super().save_module(obj, name)        
        if any(obj.__name__.startswith(excluded_module) for excluded_module in exclude_modules_list):
                return  # Skip pickling this module
        else:
            print(f'ModuleExcludesPickler.save_module(...): obj.__name__: {obj.__name__}')
            super().save_module(obj, name)
                
    # def save(self, obj, save_persistent_id=True):
    #     # Exclude specific types by name
    #     obj_type_name = type(obj).__name__
    #     if obj_type_name in exclude_type_names:
    #         return  # Skip this object
    #     super().save(obj, save_persistent_id)

    def save(self, obj, save_persistent_id=True):
        # Exclude specific modules
        if isinstance(obj, types.ModuleType):
            if any(obj.__name__.startswith(mod) for mod in exclude_modules_list):
                return  # Skip this module
        # Exclude specific types by name
        elif type(obj).__name__ in exclude_type_names:
            return  # Skip this object
        # Handle functions and built-in functions
        elif isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
            try:
                super().save(obj, save_persistent_id)
            except pickle.PicklingError as e:
                print(f'WARN: ModuleExcludesPickler.save(...) encountered pickling error: {e}')
                return  # Skip if function cannot be pickled
        else:
            super().save(obj, save_persistent_id)
            

# save_module_dict


def custom_dump(obj, file, protocol=None, byref=None, fmode=None, recurse=None, **kwds):#, strictio=None):
    """
    Pickle an object to a file.

    See :func:`dumps` for keyword arguments.
    """
    from dill.settings import settings
    protocol = settings['protocol'] if protocol is None else int(protocol)
    _kwds = kwds.copy()
    _kwds.update(dict(byref=byref, fmode=fmode, recurse=recurse))
    _exclude_pickler = ModuleExcludesPickler(file, protocol, **_kwds)
    _exclude_pickler.dump(obj)
    return

def custom_dumps(obj, protocol=None, byref=None, fmode=None, recurse=None, **kwds):#, strictio=None):
    """
    Pickle an object to a string.

    *protocol* is the pickler protocol, as defined for Python *pickle*.

    If *byref=True*, then dill behaves a lot more like pickle as certain
    objects (like modules) are pickled by reference as opposed to attempting
    to pickle the object itself.

    If *recurse=True*, then objects referred to in the global dictionary
    are recursively traced and pickled, instead of the default behavior
    of attempting to store the entire global dictionary. This is needed for
    functions defined via *exec()*.

    *fmode* (:const:`HANDLE_FMODE`, :const:`CONTENTS_FMODE`,
    or :const:`FILE_FMODE`) indicates how file handles will be pickled.
    For example, when pickling a data file handle for transfer to a remote
    compute service, *FILE_FMODE* will include the file contents in the
    pickle and cursor position so that a remote method can operate
    transparently on an object with an open file handle.

    Default values for keyword arguments can be set in :mod:`dill.settings`.
    """
    file = io.StringIO()
    custom_dump(obj, file, protocol, byref, fmode, recurse, **kwds)#, strictio)
    return file.getvalue()


# def custom_dump(obj):
#     with open("my_object.pkl", "wb") as file:
#         pickler = ModuleExcludesPickler(file)
#         pickler.dump(obj)



# def custom_dumps(obj):
#     with open("my_object.pkl", "wb") as file:
#         pickler = ModuleExcludesPickler(file)
#         pickler.dump(obj)
        


