import sys
import io
from types import ModuleType
from typing import List, Dict

# import pickle
# import dill
import dill as pickle

move_modules_list = {'pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper.SingleBarResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.SingleBarResult',
    'pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper.InstantaneousSpikeRateGroupsComputation':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.InstantaneousSpikeRateGroupsComputation',
    }


class RenameUnpickler(pickle.Unpickler):
	""" 
	# global_move_modules_list: Dict[str, str] - a dict with keys equal to the old full path to a class and values equal to the updated (replacement) full path to the class. Used to update the path to class definitions for loading previously pickled results after refactoring.
		Example: 	{'pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper.SingleBarResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.SingleBarResult',
					'pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper.InstantaneousSpikeRateGroupsComputation':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.InstantaneousSpikeRateGroupsComputation',
					}

	
		
	Usage:
		from pyphocorehelpers.Filesystem.pickling_helpers import RenameUnpickler, renamed_load
	

	pkl_path = curr_active_pipeline.global_computation_results_pickle_path
	with open(pkl_path, 'rb') as dbfile:
		# db = pickle.load(dbfile, **kwargs) # replace previous ` pickle.load(dbfile, **kwargs)` calls with `db = renamed_load(dbfile, **kwargs)`
		db = renamed_load(dbfile, **kwargs)
		
	"""
	def find_class(self, module, name):
		original_full_name = '.'.join((module, name))
		renamed_module = module
		assert self._move_modules_list is not None
		
		found_full_replacement_name = self._move_modules_list.get(original_full_name, None)        
		if found_full_replacement_name is not None:
			found_full_replacement_module, found_full_replacement_import_name = found_full_replacement_name.rsplit('.', 1)
			renamed_module = found_full_replacement_module
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

