import traceback

class Assert:
	""" 
	from pyphocorehelpers.assertion_helpers import Assert
		
		
	"""
	def path_exists(path):
		"""
		# Usage:
			Assert.path_exists(global_batch_result_inst_fr_file_path)
		"""
		import inspect
		
		# Get the caller's frame
		frame = inspect.currentframe().f_back
		
		# Extract the variable name from the caller's local variables
		var_name = [name for name, val in frame.f_locals.items() if val is path]
		
		# Use the first matched variable name or 'unknown' if not found
		var_name = var_name[0] if var_name else 'unknown'
		
		assert path.exists(), f"{var_name} does not exist! {var_name}: '{path}'" # Perform the assertion with detailed error message
		



