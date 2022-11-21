import configparser
import os

from .TSReader import TSReader
from .TSBenchmarkReader import TSBenchmarkReader
from .TSMultipleReader import TSMultipleReader
from .MGABReader import MGABReader
from .NABReader import NABReader
from .ODINTSReader import ODINTSReader
from .YahooS5Reader import YahooS5Reader


# get this file directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# load the package configuration file
rts_config = configparser.ConfigParser()
rts_config.read(os.path.join(dir_path, "time_series_config.ini"))
