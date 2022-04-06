

class SerializedAttributesSpecifyingClass:
    """ Implements a classmethod that returns a list of serialized_keys, allowing the implementor to specify which keys will be serialized/deserialized. 
    
        Useful for ignoring computed values, etc.
    """
    @classmethod
    def serialized_keys(cls):
        raise NotImplementedError