import abc

from .. import ICopyable


class IPipelineLayer(ICopyable):
    """The interface exposed from a layer of the pipeline.

    A layer of a pipeline must be copiable, savable, loadable and must have
    hyperparameters setters and getters (eventually empty if there are not).
    Individually, a pipeline layer must implement also at least one other
    interface. This interface is required for all layers that can be inserted in
    a pipeline object.
    """
    
    @abc.abstractmethod
    def get_input_shape(self) -> tuple:
        """Gets the input shape expected by the layer, eventually symbolic.

        Returns
        -------
        expected_input_shape : tuple
            It is the tuple representing the type of input shape that the layer
            expects as input. The tuple must be complete, considering all
            dimensions. If a dimension can be variable, it should be expressed
            with a string/letter, e.g., ("n", 5, 4) if the layer receives arrays
            with any dimensionality for axis 0 and dimension 5 and 5 for axis 1
            and 2. If two letters are identical, they represent the same value,
            e.g. ("n", "n") can be any array with two dimensions with equal
            value such as (5, 5) or (100, 100).
        """
        pass
    
    @abc.abstractmethod
    def get_output_shape(self) -> tuple:
        """Gets the output shape expected by the layer, eventually symbolic.

        Returns
        -------
        expected_output_shape : tuple
            It is the tuple representing the type of output shape that the layer
            will emit. The tuple must be complete, considering all dimensions.
            If a dimension can be variable, it should be expressed with a
            string/letter, e.g., ("n", 5, 4) if the layer receives arrays with
            any dimensionality for axis 0 and dimension 5 and 5 for axis 1 and
            2. If two letters are identical, they represent the same value, e.g.
            ("n", "n") can be any array with two dimensions with equal value
            such as (5, 5) or (100, 100).
        """
        pass
