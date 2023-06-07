import os
import sys
from pathlib import Path
from typing import List
import shutil # for _backup_extant_file(...)
from datetime import datetime
import pandas as pd
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



@function_attributes(short_name=None, tags=['filesystem','metadata','creation_time','modification_time','datetime','files'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-07 02:16', related_items=[])
def get_file_metadata(paths) -> pd.DataFrame:
    """
    Get the metadata (modification time, creation time, and file size) for each file specified by a list of pathlib.Path objects.
    :param paths: A list of pathlib.Path objects representing the file paths.
    :return: A pandas DataFrame with columns for path, modification time, creation time, and file size.
    """
    metadata = []

    for path in paths:
        if path.is_file():
            modified_time = os.path.getmtime(path)
            created_time = os.path.getctime(path)
            file_size = os.path.getsize(path) / (1024 ** 3)  # Convert to GB
            metadata.append({
                'path': str(path),
                'modification_time': datetime.fromtimestamp(modified_time),
                'creation_time': datetime.fromtimestamp(created_time),
                'file_size': file_size
            })

    df = pd.DataFrame(metadata)
    return df



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

