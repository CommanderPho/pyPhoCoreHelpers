from typing import Callable
import numpy as np



class CallbackSequence:
    """ Helper class to call a list of callbacks with the same argument sequentally """
    def __init__(self, callbacks_list, is_debug=False):
        self.is_debug=is_debug
        self.callbacks_list = callbacks_list

    def __call__(self, state):    
        # call the callbacks in order
        for i in np.arange(len(self.callbacks_list)):
            self.callbacks_list[i](state)
      
 

 
class CallbackWrapper(object):
    """docstring for CallbackWrapper."""
    def __init__(self, on_setup: Callable, on_update: Callable, state):
        super(CallbackWrapper, self).__init__()
        self.state = state
        self.on_setup = on_setup
        self.on_update = on_update
        self._is_setup = False
        
    def setup(self, *args):
        output = self.on_setup(*args)	
        self._is_setup = True
        return output

    def update(self, *args):
        if not self._is_setup:
            self.setup(*args)
            assert self._is_setup, "just ran self.setup(...) but self._is_setup is still False!"
        return self.on_update(*args)
    
