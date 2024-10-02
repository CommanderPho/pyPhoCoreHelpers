import traceback

class Assert:
	""" Convenince assertion helpers that print out the value that causes the assertion along with a reasonable message instead of showing nothing
	
	
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
		


	def len_equals(arr_or_list, required_length: int):
		""" Ensures the length is equal to the required_length, if it fails, it prints the actual length
		"""
		import inspect
		# Get the caller's frame
		frame = inspect.currentframe().f_back
		# Extract the variable name from the caller's local variables
		var_name = [name for name, val in frame.f_locals.items() if val is arr_or_list]
		# Use the first matched variable name or 'unknown' if not found
		var_name = var_name[0] if var_name else 'unknown'

		assert (len(arr_or_list) == required_length), f"{var_name} must be of length {required_length} but instead len({var_name}): {len(arr_or_list)}.\n{var_name}: {arr_or_list}\n" # Perform the assertion with detailed error message
