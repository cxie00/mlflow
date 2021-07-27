
class MLFeature():

    """
        Feature object that represents a feature. 
        Params:
            name (str): Name of feature
            type (str): Pandas datatype of feature
    """
    
    def __init__(self, name, type) -> None:
        self.name = name
        self.type = type
