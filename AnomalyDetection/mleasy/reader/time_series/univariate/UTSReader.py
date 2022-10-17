from mleasy.reader.time_series import TSReader


class UTSReader(TSReader):
    """A reader for univariate time series."""

    _ANOMALY_COL = "target"
    _SERIES_COL = "value"
    _TIMESTAMP_COL = "timestamp"
