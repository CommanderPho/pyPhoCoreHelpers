import os
import sys
import subprocess
import shutil # for _backup_extant_file(...)
import platform
from contextlib import contextmanager
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import re
from datetime import datetime, timedelta
from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes


from attrs import define, field, Factory
    

@define(slots=False)
class BaseMatchParser:
    """ 
    ## Sequential Parser:
    ### Tries a series of methods to parse a filename into a variety of formats that doesn't require nested try/catch
    ### Recieves: filename: str
    """
    def try_parse(self, filename: str) -> Optional[Dict]:
        raise NotImplementedError


@define(slots=False)
class DayDateTimeParser(BaseMatchParser):
    """ parses a generic datetime 
    """  
    def try_parse(self, filename: str) -> Optional[Dict]:
        pattern = r"(?P<export_datetime_str>.*_\d{2}\d{2}[APMF]{2})-(?P<session_str>.*)-(?P<export_file_type>\(?.+\)?)(?:_tbin-(?P<decoding_time_bin_size_str>[^)]+))"
        match = re.match(pattern, filename)
        if match is None:
            return None # failed
        
        parsed_output_dict = {}

        output_dict_keys = ['session_str', 'export_file_type', 'decoding_time_bin_size_str']

        # export_datetime_str, session_str, export_file_type = match.groups()
        export_datetime_str, session_str, export_file_type, decoding_time_bin_size_str = match.group('export_datetime_str'), match.group('session_str'), match.group('export_file_type'), match.group('decoding_time_bin_size_str')
        parsed_output_dict.update({k:match.group(k) for k in output_dict_keys})

        # Remove the leading characters that are not part of the datetime format
        cleaned_datetime_str: str = export_datetime_str.lstrip('._')

        # parse the datetime from the export_datetime_str and convert it to datetime object
        export_datetime = datetime.strptime(cleaned_datetime_str, "%Y-%m-%d_%I%M%p") # ValueError: time data '._2024-02-08_0535PM' does not match format '%Y-%m-%d_%I%M%p'
        parsed_output_dict['export_datetime'] = export_datetime

        return parsed_output_dict
    

@define(slots=False)
class DayDateOnlyParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        # day_date_only_pattern = r"(.*(?:_\d{2}\d{2}[APMF]{2})?)-(.*)-(\(.+\))"
        day_date_only_pattern = r"(\d{4}-\d{2}-\d{2})-(.*)-(\(?.+\)?)" # 
        day_date_only_match = re.match(day_date_only_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'        
        if day_date_only_match is not None:
            export_datetime_str, session_str, export_file_type = day_date_only_match.groups()
            # print(export_datetime_str, session_str, export_file_type)
            # parse the datetime from the export_datetime_str and convert it to datetime object
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")

        match = re.match(day_date_only_pattern, filename)        
        if match is None:
            return None # failed
        
        export_datetime_str, session_str, export_file_type = day_date_only_match.groups()
        output_dict_keys = ['session_str', 'export_file_type']
        parsed_output_dict = dict(zip(output_dict_keys, [session_str, export_file_type]))
        # parse the datetime from the export_datetime_str and convert it to datetime object
        export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
        parsed_output_dict['export_datetime'] = export_datetime

        return parsed_output_dict

@define(slots=False)
class DayDateWithVariantSuffixParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        # matches '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'
        day_date_with_variant_suffix_pattern = r"(?P<export_datetime_str>\d{4}-\d{2}-\d{2})[-_]?(?P<variant_suffix>[^-_]*)[-_](?P<session_str>.+?)_(?P<export_file_type>[A-Za-z_]+)"
        match = re.match(day_date_with_variant_suffix_pattern, filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26', 
        if match is None:
            return None # failed
        
        parsed_output_dict = {}
        output_dict_keys = ['session_str', 'export_file_type'] # , 'variant_suffix'
        export_datetime_str, session_str, export_file_type = match.group('export_datetime_str'), match.group('session_str'), match.group('export_file_type')
        parsed_output_dict.update({k:match.group(k) for k in output_dict_keys})
        # parse the datetime from the export_datetime_str and convert it to datetime object
        try:
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
            parsed_output_dict['export_datetime'] = export_datetime
        except ValueError as e:
            print(f'ERR: Could not parse date "{export_datetime_str}" of filename: "{filename}"') # 2024-01-18_GL_t_split_df
            return None # failed used to return ValueError when it couldn't parse, but we'd rather skip unparsable files

        return parsed_output_dict


# datetime.now().strftime("%Y%m%d%H%M%S")
# r'?(?P<datetime_str>\d{14})'

@define(slots=False)
class RoundedTimeParser(BaseMatchParser):
    def try_parse(self, filename: str) -> Optional[Dict]:
        # Define the regex pattern for matching the filename
        pattern = r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_(?P<hour>0[1-9]|1[0-2])(?P<time_separator>.+)(?P<minute>00|05|10|15|20|25|30|35|40|45|50|55)(?P<meridian>AM|PM)"
        match = re.match(pattern, filename)
        if match is None:
            return None  # pattern did not match
        
        parsed_output_dict = match.groupdict()

        # Construct the 'export_datetime' key based on the matched datetime components
        try:
            export_datetime_str = f"{parsed_output_dict['year']}-{parsed_output_dict['month']}-{parsed_output_dict['day']}_{parsed_output_dict['hour']}{parsed_output_dict['minute']}{parsed_output_dict['meridian']}"
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d_%I%M%p")
            parsed_output_dict['export_datetime'] = export_datetime
        except ValueError as e:
            print(f'ERR: Could not parse date-time string: "{export_datetime_str}"')
            return None  # datetime parsing failed

        # Optionally, remove individual components if not needed in the final output
        del parsed_output_dict['year']
        del parsed_output_dict['month']
        del parsed_output_dict['day']
        del parsed_output_dict['hour']
        del parsed_output_dict['minute']
        del parsed_output_dict['meridian']
        # Note: Depending on use case, keep or remove 'time_separator'

        return parsed_output_dict
            
        

@define(slots=False)
class AutoVersionedUniqueFilenameParser(BaseMatchParser):
    """ '20221109173951-loadedSessPickle.pkl' """
    def build_unique_filename(self, file_to_save_path, additional_postfix_extension=None) -> str:
        """ builds the filenames from the path of the form: '20221109173951-loadedSessPickle.pkl'"""
        if not isinstance(file_to_save_path, Path):
            file_to_save_path = Path(file_to_save_path)
        extensions = file_to_save_path.suffixes # e.g. ['.tar', '.gz']
        if additional_postfix_extension is not None:
            extensions.append(additional_postfix_extension)
        unique_file_name: str = f'{datetime.now().strftime("%Y%m%d%H%M%S")}-{file_to_save_path.stem}{"".join(extensions)}'
        return unique_file_name

    def try_parse(self, filename: str) -> Optional[Dict]:
        # matches '20221109173951-loadedSessPickle.pkl'
        
        # Regex pattern to match the unique file name format
        pattern = r'(?P<prefix_str>.+?)?-?(?P<datetime_str>\d{14})-(?P<stem>.+?)(?P<extensions>(\.\w+)*)$'
        match = re.match(pattern, filename)
        if match is None:
            return None # failed
        
        prefix_str = match.group("prefix_str")
        datetime_str = match.group("datetime_str")
        stem = match.group("stem")
        extensions = match.group("extensions")
        
        # parse the datetime from the datetime_str and convert it to datetime object
        try:
            datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
        except ValueError as e:
            print(f'ERR: Could not parse date "{datetime_str}" of filename: "{filename}"')
            return None # failed used to return ValueError when it couldn't parse, but we'd rather skip unparsable files

        # Separate multiple extensions if necessary
        extension_list = extensions.split(".") if extensions else []
        extension_list = ["." + ext for ext in extension_list if ext] # prepend '.' to each extension
        
        # Create a dictionary to store the parsed components
        parsed_output_dict = {
            'prefix_str': prefix_str,
            "datetime": datetime_obj,
            "stem": stem,
            "extensions": extension_list
        }
        return parsed_output_dict


@define(slots=False)
class AutoVersionedExtantFileBackupFilenameParser(BaseMatchParser):
    """ 'backup-20221109173951-loadedSessPickle.pkl.bak' """
    def build_backup_filename(self, file_to_save_path, backup_extension:str='.bak') -> str:
        """ builds the filenames from the path of the form: 'backup-20221109173951-loadedSessPickle.pkl.bak'"""
        if not isinstance(file_to_save_path, Path):
            file_to_save_path = Path(file_to_save_path)
        backup_file_name: str = f'backup-{datetime.now().strftime("%Y%m%d%H%M%S")}-{file_to_save_path.name}{backup_extension}'
        return backup_file_name

    def try_parse(self, filename: str) -> Optional[Dict]:
        # matches 'backup-20221109173951-loadedSessPickle.pkl.bak'
        
        # Regex pattern to match the unique file name format
        pattern = r'backup-(?P<datetime_str>\d{14})-(?P<stem>.+?)(?P<extensions>(\.\w+)*)$'
        match = re.match(pattern, filename)
        if match is None:
            return None # failed
        
        datetime_str = match.group("datetime_str")
        stem = match.group("stem")
        extensions = match.group("extensions")
        
        # parse the datetime from the datetime_str and convert it to datetime object
        try:
            datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
        except ValueError as e:
            print(f'ERR: Could not parse date "{datetime_str}" of filename: "{filename}"')
            return None # failed used to return ValueError when it couldn't parse, but we'd rather skip unparsable files

        # Separate multiple extensions if necessary
        extension_list = extensions.split(".") if extensions else []
        extension_list = ["." + ext for ext in extension_list if ext] # prepend '.' to each extension
        
        # Create a dictionary to store the parsed components
        parsed_output_dict = {
            "datetime": datetime_obj,
            "stem": stem,
            "extensions": extension_list
        }
        return parsed_output_dict



def try_datetime_detect_by_split(a_filename: str, split_parts_delimiter: str = '_'):
    """ tries to find a datetime-parsable component anywhere in the string after splitting by `split_parts_delimiter` 

    from pyphocorehelpers.Filesystem.path_helpers import try_datetime_detect_by_split

    parsed_output_dict, (successfully_parsed_to_date_split_filename_parts, non_date_split_filename_parts) = 
    """
    split_filename_parts = a_filename.split(split_parts_delimiter)
    day_date_pattern = r"(?P<export_datetime_str>\d{4}-\d{2}-\d{2})"
    parsed_output_dict = {}
    # non_datetime_filename_parts = []
    # valid_datetime_filename_parts = []

    successfully_parsed_to_date_split_filename_parts = []
    non_date_split_filename_parts = []

    for a_split_token in split_filename_parts:
        a_day_date_match = re.match(day_date_pattern, a_split_token) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'        
        if a_day_date_match is None:
            non_date_split_filename_parts.append(a_split_token)
            continue
        # parse the datetime from the export_datetime_str and convert it to datetime object
        try:
            export_datetime_str = a_day_date_match.group('export_datetime_str')
            export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
            # parsed_output_dict['detected_datetime'] = export_datetime
            parsed_output_dict['detected_datetime'] = export_datetime
            successfully_parsed_to_date_split_filename_parts.append(a_split_token)

        except ValueError as e:
            non_date_split_filename_parts.append(a_split_token)
            continue

    return parsed_output_dict, (successfully_parsed_to_date_split_filename_parts, non_date_split_filename_parts)


def try_detect_full_file_export_filename(a_filename: str):
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
    split_filename_parts = a_filename.split('_')
    day_date_pattern = r"(?P<export_file_type>[A-Za-z_]+)[-_]?(?P<export_datetime_str>\d{4}-\d{2}-\d{2})?[-_]?(?P<variant_suffix>[^-_.]*)"
    match = re.match(day_date_pattern, a_filename) # '2024-01-04-kdiba_gor01_one_2006-6-08_14-26'        
    if match is None:
        return None

    parsed_output_dict = {}

    export_file_type = match.group('export_file_type') 
    if export_file_type is not None:
        parsed_output_dict['export_file_type'] = export_file_type.strip('_') # .strip('_') drops the trailing underscore if it has one

    # parse the datetime from the export_datetime_str and convert it to datetime object
    try:
        export_datetime_str = match.group('export_datetime_str')
        export_datetime = datetime.strptime(export_datetime_str, "%Y-%m-%d")
        # parsed_output_dict['detected_datetime'] = export_datetime
        parsed_output_dict['detected_datetime'] = export_datetime
    except (ValueError, TypeError) as e:
        pass

    variant_suffix = match.group('variant_suffix') 
    if (variant_suffix is not None) and (len(variant_suffix) > 0):
        parsed_output_dict['variant_suffix'] = variant_suffix

    return parsed_output_dict



## INPUTS: basename
@function_attributes(short_name=None, tags=['parse', 'filename'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-28 10:10', related_items=[])
def try_parse_chain(basename: str, debug_print:bool=False):
    """ tries to parse the basename with the list of parsers. 
    
    Usage:
    
        from pyphocorehelpers.Filesystem.path_helpers import try_parse_chain
    
        basename: str = _test_h5_filename.stem
        final_parsed_output_dict = try_parse_chain(basename=basename)
        final_parsed_output_dict

    """
    # _filename_parsers_list = (DayDateTimeParser(), DayDateWithVariantSuffixParser(), DayDateOnlyParser())
    _filename_parsers_list = (AutoVersionedExtantFileBackupFilenameParser(), AutoVersionedUniqueFilenameParser(), DayDateTimeParser(), DayDateOnlyParser(), DayDateWithVariantSuffixParser())
    final_parsed_output_dict = None
    for a_test_parser in _filename_parsers_list:
        a_parsed_output_dict = a_test_parser.try_parse(basename)
        if a_parsed_output_dict is not None:
            ## best parser, stop here
            if debug_print:
                print(f'got parsed output {a_test_parser} - result: {a_parsed_output_dict}, basename: {basename}')
            final_parsed_output_dict = a_parsed_output_dict
            return final_parsed_output_dict
        
    return final_parsed_output_dict

# ==================================================================================================================== #
# End Parsers                                                                                                          #
# ==================================================================================================================== #


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
    unique_file_name: str = AutoVersionedUniqueFilenameParser().build_unique_filename(file_to_save_path, additional_postfix_extension=additional_postfix_extension)
    unique_save_path: Path = parent_path.joinpath(unique_file_name)
    # print(f"'{file_to_save_path}' backing up -> to_file: '{unique_save_path}'")
    return unique_save_path, unique_file_name


def parse_unique_file_name(unique_file_name: str):
    """ reciprocal to parse filenames created with `build_unique_filename`

    Usage:

    from pyphocorehelpers.Filesystem.path_helpers import parse_unique_file_name


    """
    a_parser = AutoVersionedUniqueFilenameParser()
    return a_parser.try_parse(unique_file_name)


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
    # backup_file_name = f'backup-{datetime.now().strftime("%Y%m%d%H%M%S")}-{file_to_backup_path.name}{backup_extension}'
    backup_file_name: str = AutoVersionedExtantFileBackupFilenameParser().build_backup_filename(file_to_backup_path, backup_extension=backup_extension)
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

    platform_str: str = _get_platform_str()
    if platform_str == 'darwin':
        subprocess.call(('open', filename))
    elif platform_str in ['win64', 'win32', 'windows']:
        # os.startfile(filename.replace('/', '\\'))
        os.startfile(filename.replace('/', '\\'), 'open')
        # import webbrowser
        # webbrowser.open(filename.replace('/', '\\'))

    elif platform_str == 'wsl':
        subprocess.call('cmd.exe /C start'.split() + [filename])
    elif platform_str == 'linux':
        subprocess.call(('xdg-open', filename))
    else:
        raise ValueError(f"Unsupported platform: {platform_str}")




def sanitize_filename_for_Windows(original_proposed_filename: str) -> str:
    """ 2024-04-28 - sanitizes a proposed filename such that it is valid for saving (in Windows). 

    Currently it only replaces the colon (":") with a "-". Can add more forbidden characters and their replacements to `file_sep_replace_dict` as I discover/need them

    Usage:
        from pyphocorehelpers.Filesystem.path_helpers import sanitize_filename_for_Windows

        original_proposed_filename: str = "wcorr_diff_Across Sessions 'wcorr_diff' (8 Sessions) - time bin size: 0.025 sec"
        good_filename: str = sanitize_filename_for_Windows(original_proposed_filename)
        good_filename
    
    """
    file_sep_replace_dict = {":":"-", "?":"X"}
    refined_filename: str = original_proposed_filename
    for k, v in file_sep_replace_dict.items():
        refined_filename = refined_filename.replace(k, v)
    return refined_filename