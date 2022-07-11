import os    
import subprocess
from sys import platform
from warnings import warn
if platform == "linux" or platform == "linux2":
    # linux
    pass
elif platform == "darwin":
    # OS X
    pass
elif platform == "win32":
    # Windows...
    FILEBROWSER_PATH = os.path.join(os.getenv('WINDIR'), 'explorer.exe')
    def reveal_in_windows_explorer(path):
        # explorer would choke on forward slashes
        path = os.path.normpath(path)
        if os.path.isdir(path):
            subprocess.run([FILEBROWSER_PATH, path])
        elif os.path.isfile(path):
            subprocess.run([FILEBROWSER_PATH, '/select,', path])
            




def reveal_in_system_file_manager(path):
    """ call to reveal a file in the system file browser. Currently only known to work on windows
    
    Usage:
        from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager
        reveal_in_system_file_manager(r'R:\data\Output\2022-07-11\2006-6-07_11-26-53\maze2')
    """
    if platform == "win32":
        reveal_in_windows_explorer(path)
        return
    elif platform == "linux" or platform == "linux2":
        # linux
        pass
    elif platform == "darwin":
        # OS X
        pass
    
    warn('reveal_in_system_file_manager(...) is currently not supported on platform "{platform}", known supported platforms: ["Windows"]')
        
## Now you can see the current PDF output result by:
# reveal_in_windows_explorer(curr_pdf_save_path)