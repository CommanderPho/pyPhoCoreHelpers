from pathlib import Path
from typing import List
import shutil # for _backup_extant_file(...)
from datetime import datetime

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

