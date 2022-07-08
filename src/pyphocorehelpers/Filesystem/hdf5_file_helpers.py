import h5py

ignored_keys = ['#subsystem#', '#refs#']


def h5dump(path, group='/', enable_print_attributes=False):
    """ print HDF5 file metadata
    group: you can give a specific group, defaults to the root group                 
    """
    with h5py.File(path,'r') as f:
         HDF5_Helper.descend_obj(f[group], enable_print_attributes=enable_print_attributes)
         
class HDF5_Helper(object):
    """docstring for hdf5Helper."""
    
    def __init__(self, path, group='/'):
        super(HDF5_Helper, self).__init__()
        self.path = path
        self.group = group
        self.output_tree = {}
        self.perform_enumerate()


    def perform_descend_obj(self, obj, sep='\t', enable_print_attributes=False, debug_print=False):
        """ Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
        """
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            obj_keys = [a_key for a_key in obj.keys() if '#' not in a_key]
            for key in obj_keys:
                print(sep, '-', key, ':', obj[key])
                self.perform_descend_obj(obj[key], sep=sep+'\t', enable_print_attributes=enable_print_attributes, debug_print=debug_print)
        elif type(obj)==h5py._hl.dataset.Dataset:
            if enable_print_attributes:
                obj_keys = [a_key for a_key in obj.attrs.keys() if '#' not in a_key]
                for key in obj_keys:
                    # Do not descend any further, just print the attributes
                    print(sep+'\t', '-', key, ':', obj.attrs[key])
                    
    def perform_enumerate(self, debug_print=False):
        self.output_tree = {}
        with h5py.File(self.path,'r') as f:
            self.perform_descend_obj(f[group], enable_print_attributes=enable_print_attributes, debug_print=debug_print)

        
        
    @classmethod
    def descend_obj(cls, obj, sep='\t', enable_print_attributes=False):
        """
            Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
        """
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            obj_keys = [a_key for a_key in obj.keys() if '#' not in a_key]
            for key in obj_keys:
                print(sep, '-', key, ':', obj[key]) # - behavioral_epochs : <HDF5 dataset "behavioral_epochs": shape (1, 6), type "<u4">
                cls.descend_obj(obj[key], sep=sep+'\t', enable_print_attributes=enable_print_attributes)
        elif type(obj)==h5py._hl.dataset.Dataset:
            if enable_print_attributes:
                obj_keys = [a_key for a_key in obj.attrs.keys() if '#' not in a_key]
                for key in obj_keys:
                    # Do not descend any further, just print the attributes
                    print(sep+'\t', '-', key, ':', obj.attrs[key])
            
    @classmethod
    def h5dump(cls, path, group='/', enable_print_attributes=False):
        """ print HDF5 file metadata

        group: you can give a specific group, defaults to the root group
        
            ## For enable_print_attributes == False:
                    - behavioral_epochs : <HDF5 dataset "behavioral_epochs": shape (1, 6), type "<u4">
                    - behavioral_periods_table : <HDF5 dataset "behavioral_periods_table": shape (1, 6), type "<u4">
                    - definitions : <HDF5 group "/active_processing/definitions" (3 members)>
                    - behavioral_epoch : <HDF5 group "/active_processing/definitions/behavioral_epoch" (2 members)>
                            - classNames : <HDF5 dataset "classNames": shape (3, 1), type "|O">
                            - classValues : <HDF5 dataset "classValues": shape (3, 1), type "<f8">
                    - behavioral_state : <HDF5 group "/active_processing/definitions/behavioral_state" (2 members)>
                            - classNames : <HDF5 dataset "classNames": shape (4, 1), type "|O">
                            - classValues : <HDF5 dataset "classValues": shape (4, 1), type "<f8">
                    - speculated_unit_info : <HDF5 group "/active_processing/definitions/speculated_unit_info" (2 members)>
                            - classCutoffValues : <HDF5 dataset "classCutoffValues": shape (4, 1), type "<f8">
                            - classNames : <HDF5 dataset "classNames": shape (3, 1), type "|O">
                    - earliest_start_timestamp : <HDF5 dataset "earliest_start_timestamp": shape (1, 1), type "<f8">
                    - position_table : <HDF5 dataset "position_table": shape (1, 6), type "<u4">
                    - processed_array : <HDF5 dataset "processed_array": shape (1, 2), type "|O">
                    - speed_table : <HDF5 dataset "speed_table": shape (1, 6), type "<u4">
                    - spikes : <HDF5 dataset "spikes": shape (1, 6), type "<u4">

            ## For enable_print_attributes == True:
                    - behavioral_epochs : <HDF5 dataset "behavioral_epochs": shape (1, 6), type "<u4">
                            - H5PATH : b'/active_processing'
                            - MATLAB_class : b'table'
                            - MATLAB_object_decode : 3
                    - behavioral_periods_table : <HDF5 dataset "behavioral_periods_table": shape (1, 6), type "<u4">
                            - H5PATH : b'/active_processing'
                            - MATLAB_class : b'table'
                            - MATLAB_object_decode : 3
                    - definitions : <HDF5 group "/active_processing/definitions" (3 members)>
                    - behavioral_epoch : <HDF5 group "/active_processing/definitions/behavioral_epoch" (2 members)>
                            - classNames : <HDF5 dataset "classNames": shape (3, 1), type "|O">
                                    - H5PATH : b'/active_processingdefinitionsbehavioral_epoch'
                                    - MATLAB_class : b'cell'
                            - classValues : <HDF5 dataset "classValues": shape (3, 1), type "<f8">
                                    - H5PATH : b'/active_processingdefinitionsbehavioral_epoch'
                                    - MATLAB_class : b'double'
                    - behavioral_state : <HDF5 group "/active_processing/definitions/behavioral_state" (2 members)>
                            - classNames : <HDF5 dataset "classNames": shape (4, 1), type "|O">
                                    - H5PATH : b'/active_processingdefinitionsbehavioral_state'
                                    - MATLAB_class : b'cell'
                            - classValues : <HDF5 dataset "classValues": shape (4, 1), type "<f8">
                                    - H5PATH : b'/active_processingdefinitionsbehavioral_state'
                                    - MATLAB_class : b'double'
                    - speculated_unit_info : <HDF5 group "/active_processing/definitions/speculated_unit_info" (2 members)>
                            - classCutoffValues : <HDF5 dataset "classCutoffValues": shape (4, 1), type "<f8">
                                    - H5PATH : b'/active_processingdefinitionsspeculated_unit_info'
                                    - MATLAB_class : b'double'
                            - classNames : <HDF5 dataset "classNames": shape (3, 1), type "|O">
                                    - H5PATH : b'/active_processingdefinitionsspeculated_unit_info'
                                    - MATLAB_class : b'cell'
                    - earliest_start_timestamp : <HDF5 dataset "earliest_start_timestamp": shape (1, 1), type "<f8">
                            - H5PATH : b'/active_processing'
                            - MATLAB_class : b'double'
                    - position_table : <HDF5 dataset "position_table": shape (1, 6), type "<u4">
                            - H5PATH : b'/active_processing'
                            - MATLAB_class : b'table'
                            - MATLAB_object_decode : 3
                    - processed_array : <HDF5 dataset "processed_array": shape (1, 2), type "|O">
                            - H5PATH : b'/active_processing'
                            - MATLAB_class : b'cell'
                    - speed_table : <HDF5 dataset "speed_table": shape (1, 6), type "<u4">
                            - H5PATH : b'/active_processing'
                            - MATLAB_class : b'table'
                            - MATLAB_object_decode : 3
                    - spikes : <HDF5 dataset "spikes": shape (1, 6), type "<u4">
                            - H5PATH : b'/active_processing'
                            - MATLAB_class : b'table'
                            - MATLAB_object_decode : 3
                    
        """
        with h5py.File(path,'r') as f:
            cls.descend_obj(f[group], enable_print_attributes=enable_print_attributes)
            
# def descend_obj(obj, sep='\t'):
#     """
#     Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
#     """
#     if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
#         for key in obj.keys():
#             print(sep,'-',key,':',obj[key])
#             descend_obj(obj[key],sep=sep+'\t')
#     elif type(obj)==h5py._hl.dataset.Dataset:
#         for key in obj.attrs.keys():
#             print(sep+'\t','-',key,':',obj.attrs[key])

         
