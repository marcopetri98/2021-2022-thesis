import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from setup_datasets import create_path_if_not_exist, read_in_chunks, \
    extract_and_get_new_filename

_SOTA_METHODS = [
    "anomalytransformer",
    "coca",
    "couta",
    "dghl",
    "gta",
    "inrad",
    "nsbif",
    "omnianomaly",
    "slmr",
    "tadgan",
    "tfad",
    "tranad"
]
_DEFAULT_FOLDER = "./sota_approaches"
_METHODS_LOCATIONS = {
    "anomalytransformer": "https://api.github.com/repos/thuml/Anomaly-Transformer/zipball",
    "coca": "https://api.github.com/repos/ruiking04/coca/zipball",
    "couta": "https://api.github.com/repos/xuhongzuo/couta/zipball",
    "dghl": "https://api.github.com/repos/cchallu/dghl/zipball",
    "gta": "https://api.github.com/repos/ZEKAICHEN/GTA/zipball",
    "inrad": "https://api.github.com/repos/KyeongJoong/INRAD/zipball",
    "nsbif": "https://api.github.com/repos/NSIBF/NSIBF/zipball",
    "omnianomaly": "https://api.github.com/repos/NetManAIOps/OmniAnomaly/zipball",
    "slmr": "https://api.github.com/repos/qiumiao30/SLMR/zipball",
    "tadgan": "https://api.github.com/repos/sintel-dev/Orion/zipball",
    "tfad": "https://api.github.com/repos/damo-di-ml/cikm22-tfad/zipball",
    "tranad": "https://api.github.com/repos/imperial-qore/TranAD/zipball"
}
_CHUNK_SIZE = 8192


def download_methods(to_download: str | list[str] = "all",
                     destination_folder: str = _DEFAULT_FOLDER) -> None:
    """Downloads the methods repositories.
    
    Parameters
    ----------
    to_download : str or list[str], default="all"
        It is the list of the methods to download.
    
    destination_folder : str, default=_DEFAULT_FOLDER
        It is the default folder in which the methods must be saved.

    Returns
    -------
    None
    """
    create_path_if_not_exist(destination_folder,
                             "Successfully accessed the download folder")
    
    if to_download == "all":
        to_download = set(_SOTA_METHODS)
        
    for method in to_download:
        print("Start to download the repository of", method)
        
        repo_link = _METHODS_LOCATIONS[method]

        req = urllib.request.Request(repo_link,
                                     headers={"accept": "application/vnd.github+json"})

        # download the repository
        downloaded_zip = Path(destination_folder, method + ".zip")
        method_folder = Path(destination_folder, method)
        create_path_if_not_exist(str(method_folder), "Successfully accessed method's folder")
        with urllib.request.urlopen(req) as f:
            with open(str(downloaded_zip), "wb") as out:
                read_in_chunks(f, out, _CHUNK_SIZE, end="\n")

        # extract the repository
        extract_and_get_new_filename(downloaded_zip,
                                     method_folder,
                                     method)
        
        # extract contents in the folder
        folder_path = Path(method_folder, str(os.listdir(method_folder)[0]))
        for file_or_dir_path in folder_path.glob("*"):
            shutil.move(str(file_or_dir_path), method_folder)
        shutil.rmtree(folder_path)
        
        # remove the zip
        os.remove(str(downloaded_zip))
        
        print()


def process_arguments(argv) -> dict:
    """Process arguments passed to the script.

    Parameters
    ----------
    argv
        Arguments passed to the script when it is run as main.

    Returns
    -------
    options : dict
        It is a dictionary containing all the options of the script, if they are
        not passed to the program they have None as value.
    """
    parser = argparse.ArgumentParser(prog="setup-repos.py",
                                     description="Downloads and setup all the "
                                                 "used repos with state-of-the-art "
                                                 "methods.")
    parser.add_argument("-m", "--methods",
                        action="extend",
                        nargs="+",
                        help="defines which state-of-the-art methods that have "
                             "been used for comparison must be downloaded to "
                             "their respective location",
                        choices=set(_SOTA_METHODS))
    parser.add_argument("-f", "--folder",
                        help="specifies the location of the folder where the "
                             "user wants to download the state-of-the-art "
                             "methods. To be precise, the folder containing "
                             "all the other folders of all methods")

    output = parser.parse_args(argv[1:])
    arguments = vars(output)
    
    if arguments["folder"] is not None and not os.path.exists(arguments["folder"]):
        parser.error("The folder must be a valid path")
    
    return arguments


if __name__ == "__main__":
    options = process_arguments(sys.argv)
    
    if options["methods"] is None:
        options["methods"] = "all"
        
    if options["folder"] is None:
        options["folder"] = _DEFAULT_FOLDER
        
    download_methods(options["methods"],
                     options["folder"])
    