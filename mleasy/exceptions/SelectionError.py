

class SelectionError(ValueError):
    """Error raised when a variable does not take a value from a fixed list.
    """
    def __init__(self, values, val):
        self.values = values
        self.val = val
        
        super().__init__(f"Expected one of {values}, got {val}")
