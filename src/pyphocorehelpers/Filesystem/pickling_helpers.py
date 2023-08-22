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


# class CustomUnpickler(pickle.Unpickler):
#     """python's Unpickler extended to interpreter sessions and more types"""
#     from .settings import settings
#     _session = False

#     def find_class(self, module, name):
#         if (module, name) == ('__builtin__', '__main__'):
#             return self._main.__dict__ #XXX: above set w/save_module_dict
#         elif (module, name) == ('__builtin__', 'NoneType'):
#             return type(None) #XXX: special case: NoneType missing
#         if module == 'dill.dill': module = 'dill._dill'
#         return pickle.Unpickler.find_class(self, module, name)

#     def __init__(self, *args, **kwds):
#         settings = pickle.Pickler.settings
#         _ignore = kwds.pop('ignore', None)
#         pickle.Unpickler.__init__(self, *args, **kwds)
#         self._main = _main_module
#         self._ignore = settings['ignore'] if _ignore is None else _ignore


class RenameUnpickler(pickle.Unpickler):
	""" 

	from pyphocorehelpers.Filesystem.pickling_helpers import RenameUnpickler
	RenameUnpickler.move_modules_list = {'pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper.SingleBarResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.SingleBarResult',
	'pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper.InstantaneousSpikeRateGroupsComputation':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.InstantaneousSpikeRateGroupsComputation',
	}


	pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.InstantaneousSpikeRateGroupsComputation

	{'pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper.SingleBarResult':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.SingleBarResult',
	'pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper.InstantaneousSpikeRateGroupsComputation':'pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations.InstantaneousSpikeRateGroupsComputation',
	}


	kwargs = {}
	pkl_path = curr_active_pipeline.global_computation_results_pickle_path
	with open(pkl_path, 'rb') as dbfile:
		# db = pickle.load(dbfile, **kwargs)
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

def renamed_load(file_obj, **kwargs):
	""" from pyphocorehelpers.Filesystem.pickling_helpers import renamed_load """
	return RenameUnpickler(file_obj, **kwargs).load()

# def renamed_loads(pickled_bytes):
#     file_obj = io.BytesIO(pickled_bytes)
#     return renamed_load(file_obj)



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

