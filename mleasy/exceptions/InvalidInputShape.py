

class InvalidInputShape(Exception):
    """An exception thrown if the input array has invalid shape.
    """
    def __init__(self, expected_shape: tuple,
                 shape: tuple):
        self.expected_shape = expected_shape
        self.shape = shape
        
        super().__init__(f"Received shape {expected_shape} with expected shape {shape}")
