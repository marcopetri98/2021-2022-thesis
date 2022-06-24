from reader.time_series.TSReader import TSReader


class UTSReader(TSReader):
    """A reader for univariate time series."""

    _ANOMALY_COL = "target"
    _SERIES_COL = "value"
    _TIMESTAMP_COL = "timestamp"
