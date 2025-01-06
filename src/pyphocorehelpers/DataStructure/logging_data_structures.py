from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from attrs import define, field, Factory
import neuropy.utils.type_aliases as types
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

# from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
# from qtpy import QtCore, QtWidgets
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore


@define(repr=False, slots=False)
class LoggingBaseClass(QtCore.QObject):
	""" 
	
	Used By: Spike3DRasterBottomPlaybackControlBar, 
	
	from pyphocorehelpers.DataStructure.logging_data_structures import LoggingBaseClass
	
	logger = LoggingBaseClass(log_records=[])
	logger.sigLogUpdated.connect(self.on_log_updated)
	
	
	"""
	log_records: List[str] = field(default=Factory(list))
	

	sigLogUpdated = QtCore.pyqtSignal(object) # passes self
	

	@property
	def flattened_log_text(self) -> str:
		"""The flattened_log_text property."""
		return self.get_flattened_log_text(flattening_delimiter='\n')
									 

	def get_flattened_log_text(self, flattening_delimiter: str='\n', limit_to_n_most_recent: Optional[int]=None) -> str:
		"""The flattened_log_text property."""
		if (limit_to_n_most_recent is not None) and (len(self.log_records) > limit_to_n_most_recent):
			active_log_records = self.log_records[-limit_to_n_most_recent:]
		else:
			active_log_records = self.log_records
		return flattening_delimiter.join(active_log_records)
		
	
	def __attrs_pre_init__(self):
		# super().__init__(parent=None) # normally have to do: super(ToggleButtonModel, self).__init__(parent)
		QtCore.QObject.__init__(self) # some inheritors of QObject seem to do this instead
		# note that the use of super() is often avoided because Qt does not allow to inherit from multiple QObject subclasses.

	# def __init__(self):
	# 	QtCore.QObject.__init__(self)

	def add_log_line(self, new_line: str, allow_split_newlines: bool = True):
		""" adds an additional entry to the log """
		## start by splitting on any newlines
		if allow_split_newlines:
			new_lines = new_line.splitlines()
			self.log_records.extend(new_lines)
		else:
			self.log_records.append(new_line)
			
	def add_log_lines(self, new_lines: List[str], allow_split_newlines: bool = True):
		""" adds an additional entries to the log """
		for a_line in new_lines:
			self.add_log_line(new_line=a_line, allow_split_newlines=allow_split_newlines)
					