from math import sin, pi, cos
from typing import Callable


class Seasonality(object):
    """Class defining the seasonality component for a time series.

    Attributes
    ----------
    type : str
        It defines the type os seasonal component. It can be one of: "custom",
        "sine", "cosine", "positive_triangular", "negative_triangular" and
        "triangular".

    amplitude : float
        Amplitude of the seasonal component. E.g., if the amplitude is 10 a sine
        seasonality will vary from -5 to 5.

    seasonality_length : float
        Length in seconds of the seasonal components

    seasonality_func : Callable
        If the function representing the seasonality in case the type os the
        seasonality is custom.

    Notes
    -----
    It is important to remember that the seasonality functions must be periodic,
    and they require an input varying from 0 to 1 representing how much they're
    close to the start or the end of the period.
    """
    ALLOWED_SEASONALITY = ["custom",
                           "sine",
                           "cosine",
                           "positive_triangular",
                           "negative_triangular",
                           "triangular"]

    def __init__(self, type: str,
                 amplitude: float,
                 seasonality_length: float,
                 seasonality_func: Callable[[float], None] = None):
        if type not in Seasonality.ALLOWED_SEASONALITY:
            raise ValueError("The seasonality type must be one of ",
                             Seasonality.ALLOWED_SEASONALITY)

        super().__init__()
        self.type = type
        self.amplitude = amplitude
        self.length = seasonality_length
        self.seasonality_func = seasonality_func

    def compute_seasonal_value(self, elapsed_seconds: float) -> float:
        """Computes the value of the trend component of the time series.

        The seasonal component is computed by computing the position in the
        period of the seasonal function. Then, the seasonal function (i.e.
        periodic function) is called giving to it the position on its period.
        Therefore, it is possible to perfectly determine the seasonal component
        of the time series.

        Parameters
        ----------
        elapsed_seconds: float
            The number of elapsed seconds from the start of the time series.

        Returns
        -------
        seasonal_value : float
            The value of the seasonal component of the time series.
        """
        if elapsed_seconds < 0:
            raise ValueError("The elapsed time in seconds must be greater"
                             " or equal 0")

        function_input = (elapsed_seconds % self.length) / self.length

        if self.type == "custom":
            seasonality = self.seasonality_func(function_input)
        else:
            seasonality = self.__compute_seasonality(function_input)

        return seasonality

    def __compute_seasonality(self, period_position: float) -> float:
        """Performs the computation of the seasonality.

        Parameters
        ----------
        period_position : float
            The position on the period of the function.

        Returns
        -------
        seasonal : float
            The absolute seasonal value.
        """
        if period_position < 0 or period_position > 1:
            raise ValueError("The attribute period_position must be between"
                             " 0 and 1 as it represents percentage")
        seasonality_value = 0

        match self.type:
            case "sine":
                seasonality_value = (self.amplitude / 2) * sin(period_position * 2 * pi)

            case "cosine":
                seasonality_value = (self.amplitude / 2) * cos(period_position * 2 * pi)

            case "positive_triangular":
                x = period_position * self.length
                if x <= self.length/2:
                    seasonality_value = (2 * self.amplitude * x) / self.length
                else:
                    seasonality_value = 2 * self.amplitude - (2 * self.amplitude * x) / self.length

            case "negative_triangular":
                x = period_position * self.length
                if x <= self.length/2:
                    seasonality_value = - (2 * self.amplitude * x) / self.length
                else:
                    seasonality_value = -2 * self.amplitude + (2 * self.amplitude * x) / self.length

            case "triangular":
                x = period_position * self.length
                if x <= self.length/2:
                    seasonality_value = self.amplitude / 2 - (2 * self.amplitude * x) / self.length
                else:
                    seasonality_value = - 3 * self.amplitude / 2 + (2 * self.amplitude * x) / self.length

        return seasonality_value
