import os

from mleasy.reader.time_series import TSReader


class TSBenchmarkReader(TSReader):
    """A time series benchmark reader.

    Parameters
    ----------
    benchmark_location : str
        The location of the benchmark's folder.
    """

    def __init__(self, benchmark_location: str):
        super().__init__()

        self.benchmark_location = benchmark_location

        self.__check_parameters()

    def __check_parameters(self):
        """Check parameters.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If any parameter has wrong type.

        ValueError
            If any parameter has wrong value.
        """
        if not isinstance(self.benchmark_location, str):
            raise TypeError("benchmark_location must be a string")

        if not os.path.isdir(self.benchmark_location):
            raise ValueError("benchmark_location must be a directory")
