import os
from datetime import datetime


class FilesystemMetadata:
    """ helps with accessing cross-platform filesystem metadata """
    
    @staticmethod
    def get_last_modified_time(file_path: str) -> datetime:
        """
        Returns the last modified time of a file.

        :param file_path: The path to the file.
        :return: The last modified time as a datetime object.
        """
        return datetime.fromtimestamp(os.path.getmtime(file_path))

    @staticmethod
    def get_creation_time(file_path: str) -> datetime:
        """
        Returns the creation time of a file.

        :param file_path: The path to the file.
        :return: The creation time as a datetime object.
        """
        return datetime.fromtimestamp(os.path.getctime(file_path))

