import os
import sys
import subprocess
import shutil # for _backup_extant_file(...)
import platform
from contextlib import contextmanager
import pathlib
from pathlib import Path
from typing import List, Optional, Union, Dict
import re
from datetime import datetime, timedelta
from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes


def build_unique_filename(file_to_save_path, additional_postfix_extension=None):
    """ builds a unique filename for the file to be saved at file_to_save_path.

    History:
        Used to be called `from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import _build_unique_filename`

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import build_unique_filename
        unique_save_path, unique_file_name = build_unique_filename(curr_active_pipeline.pickle_path) # unique_file_name: '20221109173951-loadedSessPickle.pkl'
        unique_save_path # 'W:/Data/KDIBA/gor01/one/2006-6-09_1-22-43/20221109173951-loadedSessPickle.pkl'
    """
    if not isinstance(file_to_save_path, Path):
        file_to_save_path = Path(file_to_save_path)
    parent_path = file_to_save_path.parent # The location to store the backups in

    extensions = file_to_save_path.suffixes # e.g. ['.tar', '.gz']
    if additional_postfix_extension is not None:
        extensions.append(additional_postfix_extension)

    unique_file_name = f'{datetime.now().strftime("%Y%m%d%H%M%S")}-{file_to_save_path.stem}{"".join(extensions)}'
    unique_save_path = parent_path.joinpath(unique_file_name)
    # print(f"'{file_to_save_path}' backing up -> to_file: '{unique_save_path}'")
    return unique_save_path, unique_file_name


def parse_unique_file_name(unique_file_name):
    """ reciprocal to parse filenames created with `build_unique_filename`

    Usage:

    from pyphocorehelpers.Filesystem.path_helpers import parse_unique_file_name


    """
    # Regex pattern to match the unique file name format
    pattern = r'(?P<prefix>.+?)?-?(?P<datetime>\d{14})-(?P<stem>.+?)(?P<extensions>(\.\w+)*)$'
    match = re.match(pattern, unique_file_name)
    
    if match:
        prefix_str = match.group("prefix")
        datetime_str = match.group("datetime")
        stem = match.group("stem")
        extensions = match.group("extensions")
        
        # Parse datetime
        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
        
        # Separate multiple extensions if necessary
        extension_list = extensions.split(".") if extensions else []
        extension_list = ["." + ext for ext in extension_list if ext] # prepend '.' to each extension
        
        # Create a dictionary to store the parsed components
        parsed_components = {
            'prefix_str': prefix_str,
            "datetime": datetime_obj,
            "stem": stem,
            "extensions": extension_list
        }
        return parsed_components
    else:
        return None
        # raise ValueError("Filename does not match the expected format")
    

    

def backup_extant_file(file_to_backup_path, MAX_BACKUP_AMOUNT=2):
    """creates a backup of an existing file that would otherwise be overwritten

    Args:
        file_to_backup_path (_type_): _description_
        MAX_BACKUP_AMOUNT (int, optional):  The maximum amount of backups to have in BACKUP_DIRECTORY. Defaults to 2.
    Usage:
    from pyphocorehelpers.Filesystem.path_helpers import backup_extant_file

    """
    if not isinstance(file_to_backup_path, Path):
        file_to_backup_path = Path(file_to_backup_path).resolve()
    assert file_to_backup_path.exists(), f"file at {file_to_backup_path} must already exist to be backed-up!"
    assert (not file_to_backup_path.is_dir()), f"file at {file_to_backup_path} must be a FILE, not a directory!"
    backup_extension = '.bak' # simple '.bak' file

    backup_directory_path = file_to_backup_path.parent # The location to store the backups in
    assert file_to_backup_path.exists()  # Validate the object we are about to backup exists before we continue

    # Validate the backup directory exists and create if required
    backup_directory_path.mkdir(parents=True, exist_ok=True)

    # Get the amount of past backup zips in the backup directory already
    existing_backups = [
        x for x in backup_directory_path.iterdir()
        if x.is_file() and x.suffix == backup_extension and x.name.startswith('backup-')
    ]

    # Enforce max backups and delete oldest if there will be too many after the new backup
    oldest_to_newest_backup_by_name = list(sorted(existing_backups, key=lambda f: f.name))
    while len(oldest_to_newest_backup_by_name) >= MAX_BACKUP_AMOUNT:  # >= because we will have another soon
        backup_to_delete = oldest_to_newest_backup_by_name.pop(0)
        backup_to_delete.unlink()

    # Create zip file (for both file and folder options)
    backup_file_name = f'backup-{datetime.now().strftime("%Y%m%d%H%M%S")}-{file_to_backup_path.name}{backup_extension}'
    to_file = backup_directory_path.joinpath(backup_file_name)
    print(f"'{file_to_backup_path}' backing up -> to_file: '{to_file}'")
    shutil.copy(file_to_backup_path, to_file)
    return True
    # dest = Path('dest')
    # src = Path('src')
    # dest.write_bytes(src.read_bytes()) #for binary files
    # dest.write_text(src.read_text()) #for text files

def find_first_extant_path(path_list: List[Path]) -> Path:
    """Returns the first path in the list that exists.
    from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path

    Args:
        path_list (List[Path]): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        Path: _description_
    """
    for a_path in path_list:
        if a_path.exists():
            return a_path
    raise FileNotFoundError(f"Could not find any of the paths in the list: {path_list}")


@function_attributes(short_name=None, tags=['filesystem','find','search','discover','data','files'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-06 20:08', related_items=['print_data_files_list_as_array'])
def discover_data_files(basedir: Path, glob_pattern='*.mat', recursive=True):
    """ By default it attempts to find the all *.mat files in the root of this basedir
    Example:
        basedir: Path(r'~/data/Bapun/Day5TwoNovel')
        session_name: 'RatS-Day5TwoNovel-2020-12-04_07-55-09'

    Example 2:
        from pyphocorehelpers.Filesystem.path_helpers import discover_data_files
        
        search_parent_path = Path(r'W:\\Data\\Kdiba')
        found_autoversioned_session_pickle_files = discover_data_files(search_parent_path, glob_pattern='*-loadedSessPickle.pkl', recursive=True)
        found_global_computation_results_files = discover_data_files(search_parent_path, glob_pattern='output/*.pkl', recursive=True)

        found_files = found_global_computation_results_files + found_autoversioned_session_pickle_files
        found_files

    """
    if isinstance(basedir, str):
        basedir = Path(basedir) # convert to Path object if not already one.
    if recursive:
        glob_pattern = f"**/{glob_pattern}"
    else:
        glob_pattern = f"{glob_pattern}"
    found_files = sorted(basedir.glob(glob_pattern))
    return found_files # 'RatS-Day5TwoNovel-2020-12-04_07-55-09'


def file_uri_from_path(a_path: Union[Path, str]) -> str:
    """ returns a path as a escaped, cross-platform, and hopefully clickable uri/url string.
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path
        file_uri_from_path(r"C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2024-01-17/kdiba/gor01/one/2006-6-08_14-26-15/plot_all_epoch_bins_marginal_predictions_Laps all_epoch_binned Marginals.png")
        
    """
    if not isinstance(a_path, Path):
        a_path = Path(a_path).resolve() # we need a Path
    return a_path.as_uri() # returns a string like "file:///C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2024-01-17/kdiba/gor01/one/2006-6-08_14-26-15/plot_all_epoch_bins_marginal_predictions_Laps%20all_epoch_binned%20Marginals.png"
    



def quote_wrapped_string(a_str: str, quote_str:str="\"") -> str:
    """ takes a a_str and returns it wrapped in literal quote characters specified by `quote_str`. Defaults to double quotes 

    from pyphocorehelpers.Filesystem.path_helpers import quote_wrapped_string, unwrap_quote_wrapped_string

    """
    return f'{quote_str}{a_str}{quote_str}'

def unwrap_quote_wrapped_string(a_quote_wrapped_str: str) -> str:
    """ inverse of `quote_wrapped_string` """
    return a_quote_wrapped_str.strip().strip('\"').strip("\'")


def quote_wrapped_file_output_path_string(src_file: Path) -> str:
    """ takes a Path and returns its string representation wrapped in double quotes """
    return quote_wrapped_string(f'{str(src_file.resolve())}', quote_str='\"')

def unwrap_quote_wrapped_file_path_string(a_file_str: str) -> str:
    """ inverse of `quote_wrapped_file_output_path_string` """
    return unwrap_quote_wrapped_string(a_file_str)


@function_attributes(short_name=None, tags=['filesystem','find','search','discover','data','files'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-06 20:09', related_items=['discover_data_files'])
def print_data_files_list_as_array(filenames_list):
    """ tries to print the output list of data files from `discover_data_files` as a printable array. 
    
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import discover_data_files
        
        print_data_files_list_as_array(found_files)
    """
    # print(f'filenames_list: {filenames_list}')
    # filenames_list_str = ',\n'.join([str(a_path) for a_path in filenames_list])
    filenames_list_str = ',\n'.join([f'r"{str(a_path)}"' for a_path in filenames_list])
    print(f'filenames_list_str: [{filenames_list_str}]')
    # for a_path in filenames_list:
    #     print(f'{str(a_path)}')



def save_filelist_to_text_file(hdf5_output_paths: List[Path], filelist_path: Path, debug_print:bool=False):
    """ 
    from pyphocorehelpers.Filesystem.path_helpers import save_filelist_to_text_file
        
    _out_string, filelist_path = save_filelist_to_text_file(hdf5_output_paths, filelist_path=text_file_path, debug_print=True)

    """
    _out_string = '\n'.join([str(a_file) for a_file in hdf5_output_paths])
    if debug_print:
        print(f'{_out_string}')
        print(f'saving out to "{filelist_path}"...')
    with open(filelist_path, 'w') as f:
        f.write(_out_string)
    return _out_string, filelist_path


def read_filelist_from_text_file(filelist_path: Path, debug_print:bool=False) -> List[Path]:
    """ 
    from pyphocorehelpers.Filesystem.path_helpers import read_filelist_from_text_file
        
    read_hdf5_output_paths = read_filelist_from_text_file(filelist_path=filelist_path, debug_print=True)
    read_hdf5_output_paths

    """
    filelist: List[Path] = []
    assert filelist_path.exists()
    assert filelist_path.is_file()
    with open(filelist_path, 'r') as f:
        read_lines = f.readlines()

    assert len(read_lines) > 0
    num_file_lines: int = len(read_lines)
    if debug_print:
        print(f'num_file_lines: {num_file_lines}')


    for i, a_line in enumerate(read_lines):
        if debug_print:
            print(f'a_line[{i}]: {a_line}')
        a_file: Path = Path(unwrap_quote_wrapped_file_path_string(a_line)).resolve()
        filelist.append(a_file)

    return filelist



def save_copydict_to_text_file(file_movedict: Dict[Path,Path], filelist_path: Path, debug_print:bool=False):
    """ 

    from pyphocorehelpers.Filesystem.path_helpers import save_copydict_to_text_file
    

    """
    num_files_to_copy: int = len(file_movedict)
    operation_symbol: str = '->'
    column_separator: str = ', '

    moved_files_lines = []
    # Add header:
    moved_files_lines.append(column_separator.join(['src_file', 'operation', '_out_path']))
    for i, (src_file, _out_path) in enumerate(file_movedict.items()):
        # moved_files_lines.append(column_separator.join((f'\"{str(src_file.resolve())}\"', operation_symbol, quote_wrapped_file_output_path_string(_out_path.resolve()) )))
        moved_files_lines.append(column_separator.join((quote_wrapped_file_output_path_string(src_file.resolve()), quote_wrapped_string(operation_symbol), quote_wrapped_file_output_path_string(_out_path.resolve()) )))

    _out_string: str = '\n'.join(moved_files_lines)
    if debug_print:
        print(f'{_out_string}')
        print(f'saving out to "{filelist_path}"...')
    with open(filelist_path, 'w') as f:
        f.write(_out_string)
    return _out_string, filelist_path


def invert_filedict(file_movedict: Dict[Path,Path]) -> Dict[Path,Path]:
    """ inverts the file_movedict so the src/dest paths are flipped. """
    return {v:k for k, v in file_movedict.items()}


def read_copydict_from_text_file(filelist_path: Path, debug_print:bool=False) -> Dict[Path,Path]:
    """ 

    from pyphocorehelpers.Filesystem.path_helpers import read_copydict_from_text_file
    

    """
    # operation_symbol: str = '->'
    column_separator: str = ', '

    file_movedict: Dict[Path,Path] = {}
    assert filelist_path.exists()
    assert filelist_path.is_file()
    with open(filelist_path, 'r') as f:
        read_lines = f.readlines()

    # _out_string
    assert len(read_lines) > 0
    ## Read Header:
    header_line = read_lines.pop(0)
    if debug_print:
        print(f'header_line: {header_line}') # column_separator.join(['src_file', 'operation', '_out_path'])


    num_file_lines: int = len(read_lines)
    if debug_print:
        print(f'num_file_lines: {num_file_lines}')


    for i, a_line in enumerate(read_lines):
        if debug_print:
            print(f'a_line[{i}]: {a_line}')
        a_src_file_str, an_operator_str, an_out_path_str = a_line.split(sep=column_separator, maxsplit=3)
        an_operator: str = unwrap_quote_wrapped_string(an_operator_str)

        a_src_file: Path = Path(unwrap_quote_wrapped_file_path_string(a_src_file_str)).resolve()
        an_out_path: Path = Path(unwrap_quote_wrapped_file_path_string(an_out_path_str)).resolve()

        file_movedict[a_src_file] = an_out_path


    return file_movedict




@metadata_attributes(short_name=None, tags=['filesystem', 'helper', 'list'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-07 16:55', related_items=[])
class FileList:
    """ helpers for manipulating lists of files.
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import FileList
    """
    @staticmethod
    def excluding_pattern(paths, exclusion_pattern):
        return [str(path) for path in paths if not str(path).match(exclusion_pattern)]

    @staticmethod
    def from_set(*args) -> list[list[Path]]:
        if len(args) == 1:
            return [[Path(path) for path in a_list] for a_list in args][0]
        else:
            return [[Path(path) for path in a_list] for a_list in args]
        
    @staticmethod
    def to_set(*args) -> list[set[str]]:
        if len(args) == 1:
            return [set(str(path) for path in a_list) for a_list in args][0] # get the item so a raw `set` is returned instead of a list[set] with a single item
        else:
            return [set(str(path) for path in a_list) for a_list in args]

    @classmethod
    def subtract(cls, lhs, rhs) -> list[Path]:
        """ 
        
        Example:
            non_primary_desired_files = FileList.subtract(found_any_pickle_files, (found_default_session_pickle_files + found_global_computation_results_files))
        
        """
        return cls.from_set(cls.to_set(lhs) - cls.to_set(rhs))

    @classmethod
    def save_to_text_file(cls, paths, save_path: Path):
        return save_filelist_to_text_file(paths, filelist_path=save_path, debug_print=False)[1] # only return the path it was saved to




def convert_filelist_to_new_parent(filelist_source: List[Path], original_parent_path: Path = Path(r'/media/MAX/cloud/turbo/Data'), dest_parent_path: Path = Path(r'/media/MAX/Data')) -> List[Path]:
    """ Converts a list of file paths from their current parent, specified by `original_parent_path`, to their new parent `dest_parent_path` 

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import convert_filelist_to_new_parent
        source_parent_path = Path(r'/media/MAX/cloud/turbo/Data')
        dest_parent_path = Path(r'/media/MAX/Data')
        # # Build the destination filelist from the source_filelist and the two paths:
        filelist_dest = convert_filelist_to_new_parent(filelist_source, original_parent_path=source_parent_path, dest_parent_path=dest_parent_path)
        filelist_dest
    """
    if original_parent_path.resolve() == dest_parent_path.resolve():
        print('WARNING: convert_filelist_to_new_parent(...): no difference between original_parent_path and dest_parent_path.')
        return filelist_source
    else:
        filelist_dest = []
        for path in filelist_source:
            relative_path = str(path.relative_to(original_parent_path))
            new_path = Path(dest_parent_path) / relative_path
            filelist_dest.append(new_path)
        return filelist_dest

@function_attributes(short_name=None, tags=['path', 'root', 'search'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-07-13 14:39', related_items=[])
def find_matching_parent_path(known_paths: List[Path], target_path: Path) -> Optional[Path]:
    """ returns the matching parent path in known_paths that is a parent of target_path, otherwise returns None if no match is found. 

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import find_matching_parent_path

        known_global_data_root_parent_paths = [Path(r'W:\Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data'), Path(r'/nfs/turbo/umms-kdiba/Data')]
        prev_global_data_root_parent_path = find_matching_parent_path(known_global_data_root_parent_paths, curr_filelist[0]) # TODO: assumes all have the same root, which is a valid assumption so far. ## prev_global_data_root_parent_path should contain the matching path from the list.
        assert prev_global_data_root_parent_path is not None, f"No matching root parent path could be found!!"
        new_session_batch_basedirs = convert_filelist_to_new_parent(curr_filelist, original_parent_path=prev_global_data_root_parent_path, dest_parent_path=desired_global_data_root_parent_path)
    """
    target_path = target_path.resolve()
    for path in known_paths:
        if target_path.is_relative_to(path.resolve()):
            return path
    return None



@metadata_attributes(short_name=None, tags=['pathlib', 'path', 'Windows', 'platform_specific', 'workaround', 'PosixPath', 'posix', 'unpickling', 'pickle', 'loading'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-07-13 13:56', related_items=[])
@contextmanager
def set_posix_windows():
    """ Prevents errors unpickling POSIX Paths on Windows that were previously pickled on Unix/Linux systems
    
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows
        with set_posix_windows():
            global_batch_run = BatchRun.try_init_from_file(global_data_root_parent_path, active_global_batch_result_filename=active_global_batch_result_filename,
                                    skip_root_path_conversion=True, debug_print=debug_print) # on_needs_create_callback_fn=run_diba_batch
    """
    if platform.system() == 'Windows':
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            yield
        finally:
            pathlib.PosixPath = posix_backup
            
# ==================================================================================================================== #
# 2023-09-21 - Generating Copydicts to Move files and preserve hierarchies                                             #
# ==================================================================================================================== #
""" Usage example 2023-09-21 - Mirror Slow Data files to much faster SSD on Linux Workstation:

Usage:
    from datetime import datetime, timedelta
    from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata, get_file_metadata
    from pyphocorehelpers.Filesystem.path_helpers import discover_data_files, generate_copydict, copy_movedict, copy_file


    source_data_root = Path(r'/media/MAX/Data')
    dest_data_root = Path(r'/home/halechr/FastData')
    assert source_data_root.exists(), f"source_data_root: {source_data_root} does not exist! Is the right computer's config commented out above?"
    assert dest_data_root.exists(), f"dest_data_root: {dest_data_root} does not exist! Is the right computer's config commented out above?"

    oldest_modified_date = (datetime.now() - timedelta(days=5))

    # Find the files and build the movedicts:
    found_session_pickle_files = discover_data_files(source_data_root, glob_pattern='loadedSessPickle.pkl', recursive=True)
    found_global_computation_results_files = discover_data_files(source_data_root, glob_pattern=f'output/{completed_global_computations_filename}', recursive=True)
    file_movedict_session_pickle_files = generate_copydict(source_data_root, dest_data_root, found_files=found_session_pickle_files, only_files_newer_than=oldest_modified_date)
    file_movedict_global_computation_results_pickle_files = generate_copydict(source_data_root, dest_data_root, found_files=found_global_computation_results_files, only_files_newer_than=oldest_modified_date)

    ### Actually perform copy operations. This will take a while
    moved_files_dict_session_pickle_files = copy_movedict(file_movedict_session_pickle_files)
    moved_files_dict_global_computation_results_pickle_files = copy_movedict(file_movedict_global_computation_results_pickle_files)

"""

def copy_file(src_path: str, dest_path: str):
    """
        Copy a file from src_path to dest_path, creating any intermediate directories as needed.
        
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import copy_file
        
    """
    if not isinstance(dest_path, Path):
         dest_path = Path(dest_path).resolve()

    # Create intermediate directories if they don't exist
    dest_directory = dest_path.parent
    dest_directory.mkdir(parents=True, exist_ok=True)

    # Copy the file
    shutil.copy(src_path, str(dest_path))

    return dest_path


def copy_recursive(source_base_path, target_base_path):
    """ 
    Copy a directory tree from one location to another. This differs from shutil.copytree() that it does not
    require the target destination to not exist. This will copy the contents of one directory in to another
    existing directory without complaining.
    It will create directories if needed, but notify they already existed.
    If will overwrite files if they exist, but notify that they already existed.
    :param source_base_path: Directory
    :param target_base_path:
    :return: None
    
    Source: https://gist.github.com/NanoDano/32bb3ba25b2bd5cdf192542660ac4de0
    
    Usage:
    
        from pyphocorehelpers.Filesystem.path_helpers import copy_recursive
    
    """
    if not Path(target_base_path).exists():
        Path(target_base_path).mkdir()    
    if not Path(source_base_path).is_dir() or not Path(target_base_path).is_dir():
        raise Exception("Source and destination directory and not both directories.\nSource: %s\nTarget: %s" % ( source_base_path, target_base_path))
    for item in os.listdir(source_base_path):
        # Directory
        if os.path.isdir(os.path.join(source_base_path, item)):
            # Create destination directory if needed
            new_target_dir = os.path.join(target_base_path, item)
            try:
                os.mkdir(new_target_dir)
            except OSError:
                sys.stderr.write("WARNING: Directory already exists:\t%s\n" % new_target_dir)

            # Recurse
            new_source_dir = os.path.join(source_base_path, item)
            copy_recursive(new_source_dir, new_target_dir)
        # File
        else:
            # Copy file over
            source_name = os.path.join(source_base_path, item)
            target_name = os.path.join(target_base_path, item)
            if Path(target_name).is_file():
                sys.stderr.write("WARNING: Overwriting existing file:\t%s\n" % target_name)
            shutil.copy(source_name, target_name)
            

def generate_copydict(source_data_root, dest_data_root, found_files: list, only_files_newer_than: Optional[datetime]=None):
    """ builds a list of files to copy by filtering the found files by their modified date, and then building the destination list using `dest_data_root` 
            Compiles these values into a dict where <key: old_file_path, value: new_file_path>
    """
    # Only get files newer than date
    if only_files_newer_than is None:
        oldest_modified_date = datetime.now() - timedelta(days=5) # newer than 5 days ago
    else: 
        oldest_modified_date = only_files_newer_than

    recently_modified_source_filelist = [a_file for a_file in found_files if (FilesystemMetadata.get_last_modified_time(a_file)>oldest_modified_date)]

    # Build the destination filelist from the source_filelist and the two paths:
    filelist_dest = convert_filelist_to_new_parent(recently_modified_source_filelist, original_parent_path=source_data_root, dest_parent_path=dest_data_root)
    return dict(zip(recently_modified_source_filelist, filelist_dest))


def copy_movedict(file_movedict: dict, print_progress:bool=True) -> dict:
    """ copies each file in file_movedict from its key -> value, creating any intermediate directories as needed.
    
    """
    ## Perform the copy creating any intermediate directories as needed
    num_files_to_copy: int = len(file_movedict)
    moved_files_dict = {}
    for i, (src_file, dest_file) in enumerate(file_movedict.items()):
        if print_progress:
            print(f'copying "{src_file}"\n\t\t -> "{dest_file}"...')
        _out_path = copy_file(src_file, dest_file)
        moved_files_dict[src_file] = _out_path
        if print_progress:
            print(f'done.')
    if print_progress:
        print(f'done copying {len(moved_files_dict)} of {num_files_to_copy} files.')
    return moved_files_dict

def copy_files(filelist_source: list, filelist_dest: list) -> dict:
    """ copies each file from `filelist_source` to its corresponding destination in `filelist_dest`, creating any intermediate directories as needed.
    
    """
    return copy_movedict(dict(zip(filelist_source, filelist_dest)))



def _get_platform_str() -> str:
    """ 
    
    Usage:
        platform = _get_platform_str()
        if platform == 'darwin':
            subprocess.call(('open', filename))
        elif platform in ['win64', 'win32']:
            os.startfile(filename.replace('/','\\'))
        elif platform == 'wsl':
            subprocess.call('cmd.exe /C start'.split() + [filename])
        else:                                   # linux variants
            subprocess.call(('xdg-open', filename))

    """
    if sys.platform == 'linux':
        try:
            proc_version = open('/proc/version').read()
            if 'Microsoft' in proc_version:
                return 'wsl'
        except:
            pass
    return sys.platform


def open_file_with_system_default(filename: Union[Path, str]):
    """ attempts to open the passed file with the system default application. 

    Credit: https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-in-windows-and-mac-os

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import open_file_with_system_default

        open_file_with_system_default(r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation/InteractivePlaceCellConfig.html')

    """
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    platform = _get_platform_str()
    if platform == 'darwin':
        subprocess.call(('open', filename))
    elif platform in ['win64', 'win32']:
        os.startfile(filename.replace('/','\\'))
    elif platform == 'wsl':
        subprocess.call('cmd.exe /C start'.split() + [filename])
    else:                                   
        # linux variants
        subprocess.call(('xdg-open', filename))


