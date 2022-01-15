from collections import OrderedDict
import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..mixins.diffable import DiffableObject
from ..mixins.key_value_hashable import KeyValueHashableObject

from ..print_helpers import SimplePrintable


class UniqueCombinedConfigIdentifier(SimplePrintable, DiffableObject, KeyValueHashableObject):
    """ Note: this is not general, it's for the place cell project, and it combines a bunch of functionality. """
    def __init__(self, filter_name, active_config, variant_identifier_label=None, **kwargs):
        combined_config_dict = UniqueCombinedConfigIdentifier._build_combined_identifier(filter_name, active_config, variant_identifier_label=variant_identifier_label)
        for key, value in combined_config_dict.items():
            setattr(self, key, value) # add each of the combined items built from the separate configs

        # Dump all arguments into parameters.
        for key, value in kwargs.items():
            setattr(self, key, value)


    def items(self):
        """ Passthrough for use with DiffableObject """
        return self.__dict__.items()


    @classmethod
    def init_from_combined_dict(cls, combined_config_dict):
        return cls(**combined_config_dict)

    @staticmethod
    def _build_combined_identifier(filter_name, active_config, variant_identifier_label=None, debug_print=False):
        """ Build the combined indentifier dictionary which includes elements from the session, the function that was used to filter the session, and of course the computation config """
        combined_config_dict = {**{included_key:active_config.active_session_config.__dict__[included_key] for included_key in ['session_name']}, **active_config.computation_config.__dict__, 'filter_name':filter_name, **active_config.filter_config} # include all elements of the computation config
        if variant_identifier_label is not None:
            combined_config_dict[variant_identifier_label] = variant_identifier_label
        # combined_config_hash = hash_dictionary(combined_config_dict)
        # print(f'active_config.active_session_config: {active_config.active_session_config}')
        if debug_print:
            print(f'combined_config_dict: {combined_config_dict}\n')
        return combined_config_dict



class PhoHistoricalResultsManager(SimplePrintable):
    """ Keeps track of historical results and allows you to compare them easily. """
    def __init__(self, results_dict = OrderedDict(), outputs_dict = OrderedDict()):
        super(PhoHistoricalResultsManager, self).__init__()
        self.results = results_dict
        self.outputs = outputs_dict
        
    
    def add(self, current_config, current_results=dict(), current_outputs=dict()):
        if current_config in self.results:
            print('already exists!')
            self.results[current_config] = (self.results[current_config] | current_results) # adds/replaces the result if needed
        else:
            print('new config! adding results!')
            self.results[current_config] = current_results
            
        if current_config in self.outputs:
            print('already exists!')
            self.outputs[current_config] = (self.outputs[current_config] | current_outputs) # adds/replaces the result if needed
        else:
            print('new config! adding outputs!')
            self.outputs[current_config] = current_outputs
            
    
    
    
    
    def reset(self):
        print(f'Resetting PhoHistoricalResultsManager!')
        self.results = OrderedDict()
        self.outputs = OrderedDict()
        
        
