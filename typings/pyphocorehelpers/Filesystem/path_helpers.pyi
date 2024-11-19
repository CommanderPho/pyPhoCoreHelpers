"""
This type stub file was generated by pyright.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes
from attrs import define

@define(slots=False)
class BaseMatchParser:
    """ 
    ## Sequential Parser:
    ### Tries a series of methods to parse a filename into a variety of formats that doesn't require nested try/catch
    ### Recieves: filename: str
    """
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    
    def try_iterative_parse(self, parsed_output_dict: Dict) -> Dict:
        """ attempts to parse the parsed_output_dict 
        returns an updated version
        """
        ...
    


@define(slots=False)
class DayDateTimeParser(BaseMatchParser):
    """ parses a generic datetime 
    """
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    


@define(slots=False)
class DayDateOnlyParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    


@define(slots=False)
class DayDateWithVariantSuffixParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    


@define(slots=False)
class RoundedTimeParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    


@define(slots=False)
class AutoVersionedUniqueFilenameParser(BaseMatchParser):
    """ '20221109173951-loadedSessPickle.pkl' """
    def build_unique_filename(self, file_to_save_path, additional_postfix_extension=...) -> str:
        """ builds the filenames from the path of the form: '20221109173951-loadedSessPickle.pkl'"""
        ...
    
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    


@define(slots=False)
class AutoVersionedExtantFileBackupFilenameParser(BaseMatchParser):
    """ 'backup-20221109173951-loadedSessPickle.pkl.bak' """
    def build_backup_filename(self, file_to_save_path, backup_extension: str = ...) -> str:
        """ builds the filenames from the path of the form: 'backup-20221109173951-loadedSessPickle.pkl.bak'"""
        ...
    
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    


@define(slots=False)
class ParenWrappedDataNameParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    


@define(slots=False)
class DoubleUnderscoreSplitSessionPlusAdditionalContextParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        ...
    


def try_datetime_detect_by_split(a_filename: str, split_parts_delimiter: str = ...): # -> tuple[dict[Any, Any], tuple[list[Any], list[Any]]]:
    """ tries to find a datetime-parsable component anywhere in the string after splitting by `split_parts_delimiter` 

    from pyphocorehelpers.Filesystem.path_helpers import try_datetime_detect_by_split

    parsed_output_dict, (successfully_parsed_to_date_split_filename_parts, non_date_split_filename_parts) = 
    """
    ...

def try_detect_full_file_export_filename(a_filename: str): # -> dict[Any, Any] | None:
    """ Parses filenames like

    loadedSessPickle_test_strings = [
    'loadedSessPickle.pkl',
    'loadedSessPickle_2023-10-06.pkl',
    'loadedSessPickle_2024-03-28_Apogee.pkl',
    ]

    global_computation_results_test_strings = [
    'global_computation_results.pkl',
    'global_computation_results_2023-10-06.pkl',
    'global_computation_results_2024-03-28_Apogee.pkl',
    ]

    
    """
    ...

@function_attributes(short_name=None, tags=['parse', 'filename'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-28 10:10', related_items=[])
def try_parse_chain(basename: str, debug_print: bool = ...): # -> Dict[Any, Any] | None:
    """ tries to parse the basename with the list of parsers. 
    
    Usage:
    
        from pyphocorehelpers.Filesystem.path_helpers import try_parse_chain
    
        basename: str = _test_h5_filename.stem
        final_parsed_output_dict = try_parse_chain(basename=basename)
        final_parsed_output_dict

    """
    ...

@function_attributes(short_name=None, tags=['parse', 'filename', 'iterative'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-15 18:41', related_items=[])
def try_iterative_parse_chain(basename: str, debug_print: bool = ...): # -> dict[str, str] | Dict[Any, Any]:
    """ tries to parse the basename with the list of parsers THAT CONSUME THE INPUT STRING AS THEY PARSE IT. 
    
    Usage:
    
        from pyphocorehelpers.Filesystem.path_helpers import try_iterative_parse_chain
    
        basename: str = _test_h5_filename.stem
        final_parsed_output_dict = try_parse_chain(basename=basename)
        final_parsed_output_dict

        
    final_parsed_output_dict: {'variant_suffix': 'GL',
        'export_datetime': datetime.datetime(2024, 11, 15, 0, 0),
        'export_file_type': 'merged_complete_epoch_stats_df',
        'session_str': 'kdiba-gor01-one-2006-6-08_14-26-15',
        'custom_replay_name': 'withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]'}
        
    """
    ...

def build_unique_filename(file_to_save_path, additional_postfix_extension=...): # -> tuple[Path, str]:
    """ builds a unique filename for the file to be saved at file_to_save_path.

    History:
        Used to be called `from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import _build_unique_filename`

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import build_unique_filename
        unique_save_path, unique_file_name = build_unique_filename(curr_active_pipeline.pickle_path) # unique_file_name: '20221109173951-loadedSessPickle.pkl'
        unique_save_path # 'W:/Data/KDIBA/gor01/one/2006-6-09_1-22-43/20221109173951-loadedSessPickle.pkl'
    """
    ...

def parse_unique_file_name(unique_file_name: str): # -> Dict[Any, Any] | None:
    """ reciprocal to parse filenames created with `build_unique_filename`

    Usage:

    from pyphocorehelpers.Filesystem.path_helpers import parse_unique_file_name


    """
    ...

def backup_extant_file(file_to_backup_path, MAX_BACKUP_AMOUNT=...): # -> Literal[True]:
    """creates a backup of an existing file that would otherwise be overwritten

    Args:
        file_to_backup_path (_type_): _description_
        MAX_BACKUP_AMOUNT (int, optional):  The maximum amount of backups to have in BACKUP_DIRECTORY. Defaults to 2.
    Usage:
    from pyphocorehelpers.Filesystem.path_helpers import backup_extant_file

    """
    ...

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
    ...

@function_attributes(short_name=None, tags=['filesystem', 'find', 'search', 'discover', 'data', 'files'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-06 20:08', related_items=['print_data_files_list_as_array'])
def discover_data_files(basedir: Path, glob_pattern=..., recursive=...): # -> list[Path]:
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
    ...

def file_uri_from_path(a_path: Union[Path, str]) -> str:
    """ returns a path as a escaped, cross-platform, and hopefully clickable uri/url string.
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import file_uri_from_path
        file_uri_from_path(r"C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2024-01-17/kdiba/gor01/one/2006-6-08_14-26-15/plot_all_epoch_bins_marginal_predictions_Laps all_epoch_binned Marginals.png")
        
    """
    ...

def quote_wrapped_string(a_str: str, quote_str: str = ...) -> str:
    """ takes a a_str and returns it wrapped in literal quote characters specified by `quote_str`. Defaults to double quotes 

    from pyphocorehelpers.Filesystem.path_helpers import quote_wrapped_string, unwrap_quote_wrapped_string

    """
    ...

def unwrap_quote_wrapped_string(a_quote_wrapped_str: str) -> str:
    """ inverse of `quote_wrapped_string` """
    ...

def quote_wrapped_file_output_path_string(src_file: Path) -> str:
    """ takes a Path and returns its string representation wrapped in double quotes """
    ...

def unwrap_quote_wrapped_file_path_string(a_file_str: str) -> str:
    """ inverse of `quote_wrapped_file_output_path_string` """
    ...

@function_attributes(short_name=None, tags=['filesystem', 'find', 'search', 'discover', 'data', 'files'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-06 20:09', related_items=['discover_data_files'])
def print_data_files_list_as_array(filenames_list): # -> None:
    """ tries to print the output list of data files from `discover_data_files` as a printable array. 
    
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import discover_data_files
        
        print_data_files_list_as_array(found_files)
    """
    ...

def save_filelist_to_text_file(hdf5_output_paths: List[Path], filelist_path: Path, debug_print: bool = ...): # -> tuple[str, Path]:
    """ 
    from pyphocorehelpers.Filesystem.path_helpers import save_filelist_to_text_file
        
    _out_string, filelist_path = save_filelist_to_text_file(hdf5_output_paths, filelist_path=text_file_path, debug_print=True)

    """
    ...

def read_filelist_from_text_file(filelist_path: Path, debug_print: bool = ...) -> List[Path]:
    """ 
    from pyphocorehelpers.Filesystem.path_helpers import read_filelist_from_text_file
        
    read_hdf5_output_paths = read_filelist_from_text_file(filelist_path=filelist_path, debug_print=True)
    read_hdf5_output_paths

    """
    ...

def save_copydict_to_text_file(file_movedict: Dict[Path, Path], filelist_path: Path, debug_print: bool = ...): # -> tuple[LiteralString, Path]:
    """ 

    from pyphocorehelpers.Filesystem.path_helpers import save_copydict_to_text_file
    

    """
    ...

def invert_filedict(file_movedict: Dict[Path, Path]) -> Dict[Path, Path]:
    """ inverts the file_movedict so the src/dest paths are flipped. """
    ...

def read_copydict_from_text_file(filelist_path: Path, debug_print: bool = ...) -> Dict[Path, Path]:
    """ 

    from pyphocorehelpers.Filesystem.path_helpers import read_copydict_from_text_file
    

    """
    ...

@metadata_attributes(short_name=None, tags=['filesystem', 'helper', 'list'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-06-07 16:55', related_items=[])
class FileList:
    """ helpers for manipulating lists of files.
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import FileList
    """
    @staticmethod
    def excluding_pattern(paths, exclusion_pattern): # -> list[str]:
        ...
    
    @staticmethod
    def from_set(*args) -> list[list[Path]]:
        ...
    
    @staticmethod
    def to_set(*args) -> list[set[str]]:
        ...
    
    @classmethod
    def subtract(cls, lhs, rhs) -> list[Path]:
        """ 
        
        Example:
            non_primary_desired_files = FileList.subtract(found_any_pickle_files, (found_default_session_pickle_files + found_global_computation_results_files))
        
        """
        ...
    
    @classmethod
    def save_to_text_file(cls, paths, save_path: Path): # -> Path:
        ...
    


def convert_filelist_to_new_parent(filelist_source: List[Path], original_parent_path: Path = ..., dest_parent_path: Path = ...) -> List[Path]:
    """ Converts a list of file paths from their current parent, specified by `original_parent_path`, to their new parent `dest_parent_path` 

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import convert_filelist_to_new_parent
        source_parent_path = Path(r'/media/MAX/cloud/turbo/Data')
        dest_parent_path = Path(r'/media/MAX/Data')
        # # Build the destination filelist from the source_filelist and the two paths:
        filelist_dest = convert_filelist_to_new_parent(filelist_source, original_parent_path=source_parent_path, dest_parent_path=dest_parent_path)
        filelist_dest
    """
    ...

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
    ...

@metadata_attributes(short_name=None, tags=['pathlib', 'path', 'Windows', 'platform_specific', 'workaround', 'PosixPath', 'posix', 'unpickling', 'pickle', 'loading'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-07-13 13:56', related_items=[])
@contextmanager
def set_posix_windows(): # -> Generator[None, Any, None]:
    """ Prevents errors unpickling POSIX Paths on Windows that were previously pickled on Unix/Linux systems
    
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows
        with set_posix_windows():
            global_batch_run = BatchRun.try_init_from_file(global_data_root_parent_path, active_global_batch_result_filename=active_global_batch_result_filename,
                                    skip_root_path_conversion=True, debug_print=debug_print) # on_needs_create_callback_fn=run_diba_batch
    """
    ...

def copy_file(src_path: str, dest_path: str): # -> str | <subclass of str and Path>:
    """
        Copy a file from src_path to dest_path, creating any intermediate directories as needed.
        
    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import copy_file
        
    """
    ...

def copy_recursive(source_base_path, target_base_path): # -> None:
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
    ...

def generate_copydict(source_data_root, dest_data_root, found_files: list, only_files_newer_than: Optional[datetime] = ...): # -> dict[Any, Path]:
    """ builds a list of files to copy by filtering the found files by their modified date, and then building the destination list using `dest_data_root` 
            Compiles these values into a dict where <key: old_file_path, value: new_file_path>
    """
    ...

def copy_movedict(file_movedict: dict, print_progress: bool = ...) -> dict:
    """ copies each file in file_movedict from its key -> value, creating any intermediate directories as needed.
    
    """
    ...

def copy_files(filelist_source: list, filelist_dest: list) -> dict:
    """ copies each file from `filelist_source` to its corresponding destination in `filelist_dest`, creating any intermediate directories as needed.
    
    """
    ...

def open_file_with_system_default(filename: Union[Path, str]): # -> None:
    """ attempts to open the passed file with the system default application. 

    Credit: https://stackoverflow.com/questions/434597/open-document-with-default-os-application-in-python-both-in-windows-and-mac-os

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import open_file_with_system_default

        open_file_with_system_default(r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation/InteractivePlaceCellConfig.html')

    """
    ...

def open_vscode_link(a_vscode_link_str: str, debug_print: bool = ..., open_in_background: bool = ...): # -> None:
    """ opens the vscode link in vscode, optionally in the background to keep the calling widget focused
    
    from pyphocorehelpers.Filesystem.path_helpers import open_vscode_link

    a_vscode_link_str: str = "vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py:593"
    open_vscode_link(a_vscode_link_str=a_vscode_link_str)
    
    """
    ...

def sanitize_filename_for_Windows(original_proposed_filename: str) -> str:
    """ 2024-04-28 - sanitizes a proposed filename such that it is valid for saving (in Windows). 

    Currently it only replaces the colon (":") with a "-". Can add more forbidden characters and their replacements to `file_sep_replace_dict` as I discover/need them

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import sanitize_filename_for_Windows

        original_proposed_filename: str = "wcorr_diff_Across Sessions 'wcorr_diff' (8 Sessions) - time bin size: 0.025 sec"
        good_filename: str = sanitize_filename_for_Windows(original_proposed_filename)
        good_filename
    
    """
    ...

