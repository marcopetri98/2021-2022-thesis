import argparse
import gzip
import os
import shutil
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Iterable

import inflate64

_PUBLIC_DATASETS = ["exathlon",
                    "ghl",
                    "kitsune",
                    "mgab",
                    "nab",
                    "nasa_msl_smap",
                    "smd",
                    "ucr"]
_AVAILABLE_DATASETS = ["yahoo_s5"]
_DEFAULT_FOLDER = "C:/Users/marco/Documents/test/"
_DATASET_LOCATIONS = {
    "exathlon": "https://api.github.com/repos/exathlonbenchmark/exathlon/zipball",
    "ghl": "https://kas.pr/ics-research/dataset_ghl_1",
    "kitsune": "https://archive.ics.uci.edu/ml/machine-learning-databases/00516/",
    "mgab": "https://api.github.com/repos/MarkusThill/MGAB/zipball",
    "nab": "https://api.github.com/repos/numenta/NAB/zipball",
    "nasa_msl_smap": ["https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
                      "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"],
    "smd": "https://api.github.com/repos/NetManAIOps/OmniAnomaly/zipball",
    "ucr": "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip"
}
_MAINTAIN_FILE_DIR = {
    "nab": ["data", "labels"],
    "mgab": ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv", "10.csv"],
    "exathlon": ["data"]
}
_CHUNK_SIZE = 8192


class UCIDatasetParser(HTMLParser):
    """Reads the structure of a html page of a UCI database.

    While reading the page, the parser stores all the links of the website
    structure in a member which can be subsequently retrieved.
    """
    def __init__(self):
        super().__init__()

        self.links = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            attr_dict = {attrs[i][0]: attrs[i][1] for i in range(len(attrs))}

            if "href" in attr_dict:
                self.links.append(attr_dict["href"])

    def handle_data(self, data: str) -> None:
        if data.strip() == "Parent Directory":
            self.links.pop()


class ExtractionLoading(object):
    """A class printing loading messages to be used with threads.

    The class is intended to print loading messages for zip extraction to let
    the user know that the program is still running. The program expect an
    argument to the call representing the dataset's name.

    Attributes
    ----------
    has_to_be_stopped : bool
        States whether the application should be stopped or not.
    """
    def __init__(self):
        super().__init__()

        self.has_to_be_stopped = False

    def __call__(self, *args, **kwargs):
        words_to_print = 1
        dataset = kwargs["ds"]
        last_string = f"Waiting for {dataset} extraction..."
        white_spaces = " "*len(last_string)

        while not self.has_to_be_stopped or words_to_print < 5:
            match words_to_print:
                case 1:
                    print(white_spaces, flush=True, end="\r")
                    print("Waiting ", flush=True, end="\r")

                case 2:
                    print("Waiting for ", flush=True, end="\r")

                case 3:
                    print(f"Waiting for {dataset} ", flush=True, end="\r")

                case 4:
                    print(last_string, flush=True, end="\r")

            words_to_print += 1
            if not self.has_to_be_stopped:
                if words_to_print == 5:
                    words_to_print = 1
                sleep(1)

        print(f"The file/dataset {dataset} has been extracted")
        self.has_to_be_stopped = False


def create_path_if_not_exist(path: str, message: str = None, should_print: bool = True) -> None:
    """Create all the folders if they do not exist.

    Parameters
    ----------
    path : str
        The path to create.

    message : str, default=None
        The message to print after creating the path.

    should_print : bool, default=True
        States if the prints should be done or not.

    Returns
    -------
    None
    """
    try:
        Path(os.path.normpath(path)).mkdir(0o777)
        if should_print:
            print(message)
    except (FileNotFoundError, FileExistsError):
        if should_print:
            print(message)


def read_in_chunks(file, output, chunk_size, end: str = "\n\n") -> None:
    """Reads a file in chunk and stores it in another location.

    Parameters
    ----------
    file
        File to read.

    output
        File to write.

    chunk_size
        Dimension of the chunks.

    end : str, default="\n\n"
        The end of the print when the file has been loaded completely.

    Returns
    -------
    None
    """
    chunk = file.read(chunk_size)
    current_chunk = 1

    while len(chunk) != 0:
        output.write(chunk)
        chunk = file.read(chunk_size)
        print(f"Downloading files... Totally downloaded chunks: "
              f"{current_chunk}, Total file size: "
              f"{current_chunk * chunk_size} B", end="\r")
        current_chunk += 1

    print(f"Downloading files... Totally downloaded chunks: "
          f"{current_chunk}, Total file size: "
          f"{current_chunk * chunk_size} B", end=end)


def extract_and_get_new_filename(to_extract: str | bytes | os.PathLike,
                                 where_to_extract: str | bytes | os.PathLike,
                                 extracted_elem: str,
                                 is_archive: bool = True) -> bytes:
    """Extracts an object.

    Parameters
    ----------
    to_extract : str or bytes or PathLike
        The compressed file to decompress.

    where_to_extract : str or bytes or PathLike
        The folder in which to decompress the file.

    extracted_elem : str
        Name of the extracted element for utility printing.

    is_archive : bool, default=True
        `True` whether the object is an archive, `False` if the object is a
        single compressed file ending in `.gz`.

    Returns
    -------
    path_to_new_file : bytes
        The path to the new file.
    """
    load_printer = ExtractionLoading()
    previous_contents = os.listdir(where_to_extract)

    loading_thread = Thread(target=load_printer,
                            kwargs={"ds": extracted_elem},
                            daemon=True)
    loading_thread.start()

    if is_archive:
        shutil.unpack_archive(to_extract, where_to_extract)
    else:
        filename = os.path.normpath(to_extract).split("/")[-1]
        base_filename = filename.split(".gz")[0]
        with gzip.open(to_extract, "rb") as f_in:
            with open(os.path.join(where_to_extract, base_filename), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    load_printer.has_to_be_stopped = True
    loading_thread.join()

    new_contents = os.listdir(where_to_extract)
    new_file = list(set(new_contents).difference(previous_contents))[0]
    return os.path.normpath(os.path.join(where_to_extract, new_file))


def download_datasets(to_download: str | Iterable[str] = "all",
                      destination_folder: str | bytes | os.PathLike = _DEFAULT_FOLDER) -> None:
    """Downloads all the specified public datasets.

    Parameters
    ----------
    to_download : str or Iterable[str], default="all"
        The public datasets that are selected fow download.

    destination_folder : str or bytes or PathLike, default="./data"
        The destination folder at which the datasets must be downloaded.

    Returns
    -------
    None
    """
    create_path_if_not_exist(destination_folder,
                             "Successfully accessed the download folder")

    if to_download == "all":
        to_download = set(_PUBLIC_DATASETS)

    for dataset in to_download:
        print("Start to download the dataset", dataset)

        if dataset == "kitsune":
            links_to_visit = [_DATASET_LOCATIONS[dataset]]
            files_to_download = []

            dataset_folder = os.path.normpath(os.path.join(destination_folder, dataset))
            create_path_if_not_exist(dataset_folder,
                                     f"Created folder for {dataset}")
            print("Crawling UCI dataset's page")

            while len(links_to_visit) != 0:
                link = links_to_visit.pop()
                req = urllib.request.Request(link)

                with urllib.request.urlopen(req) as f:
                    content = f.read().decode("UTF-8")
                    parser = UCIDatasetParser()
                    parser.feed(content)
                    found_links = parser.links

                    for new_link in found_links:
                        if "." in new_link:
                            files_to_download.append(link + new_link)
                        else:
                            links_to_visit.append(link + new_link)

            print(f"There are {len(files_to_download)} files to download")
            print("Start to download files and rebuild the zip on pc")

            for file in files_to_download:
                req = urllib.request.Request(file)
                added_path = file.split(_DATASET_LOCATIONS[dataset])[1].lower()
                added_path = added_path.replace("%20", "_")
                file_path = os.path.normpath(os.path.join(dataset_folder, added_path))
                create_path_if_not_exist(str(Path(file_path).parent), should_print=False)

                with urllib.request.urlopen(req) as f:
                    with open(file_path, "wb") as out:
                        print(f"Downloading file: {added_path}")
                        read_in_chunks(f, out, _CHUNK_SIZE, end="\n")

            print()

            temp_folder = os.path.normpath(os.path.join(destination_folder, dataset + "_temp"))
            create_path_if_not_exist(temp_folder, should_print=False)
            shutil.move(dataset_folder, temp_folder)
            os.rename(temp_folder, dataset_folder)

            shutil.make_archive(dataset_folder, "zip", dataset_folder)
            shutil.rmtree(dataset_folder)
        else:
            if dataset == "nasa_msl_smap":
                dataset_link = _DATASET_LOCATIONS[dataset][0]
            else:
                dataset_link = _DATASET_LOCATIONS[dataset]

            req = urllib.request.Request(dataset_link,
                                         headers={"accept": "application/vnd.github+json"})

            with urllib.request.urlopen(req) as f:
                with open(os.path.join(destination_folder, dataset + ".zip"), "wb") as out:
                    read_in_chunks(f, out, _CHUNK_SIZE, end="\n")

            if dataset == "nasa_msl_smap":
                req = urllib.request.Request(_DATASET_LOCATIONS[dataset][1],
                                             headers={"accept": "application/vnd.github+json"})
                filename = _DATASET_LOCATIONS[dataset][1].split("/")[-1]

                with urllib.request.urlopen(req) as f:
                    with open(os.path.join(destination_folder, filename), "wb") as out:
                        content = f.read()
                        out.write(content)

                print("Building the complete archive... Please wait few seconds...")

                shutil.unpack_archive(os.path.join(destination_folder, dataset + ".zip"),
                                      os.path.join(destination_folder, dataset))
                os.remove(os.path.join(destination_folder, dataset + ".zip"))
                shutil.move(os.path.join(destination_folder, filename),
                            os.path.join(destination_folder,
                                         dataset,
                                         os.listdir(os.path.join(destination_folder, dataset))[0]))
                shutil.make_archive(os.path.join(destination_folder, dataset),
                                    "zip",
                                    os.path.join(destination_folder, dataset))
                shutil.rmtree(os.path.join(destination_folder, dataset))

                print()


def relocate_datasets(to_relocate: str | Iterable[str] = "all",
                      download_folder: str | bytes | os.PathLike = _DEFAULT_FOLDER,
                      destination_folder: str | bytes | os.PathLike = _DEFAULT_FOLDER) -> None:
    """Extracts and relocate datasets

    Parameters
    ----------
    to_relocate : str or Iterable[str], default="all"
        The datasets that the script must extract and relocate to the final
        destination folder. The datasets to relocate might be either public or
        accessible. Please, note that once you downloaded an accessible dataset
        you must move it in the folder with other datasets without renaming it.

    download_folder : str or bytes or PathLike, default="./data"
        The folder in which the datasets have been downloaded.

    destination_folder : str or bytes or PathLike, default="./data"
        The destination folder at which the datasets must be downloaded.

    Returns
    -------
    None
    """
    create_path_if_not_exist(destination_folder,
                             "Successfully accessed the download folder")

    if to_relocate == "all":
        to_relocate = set(_PUBLIC_DATASETS).union(_AVAILABLE_DATASETS)

    for dataset in to_relocate:
        dw_folder = os.path.normpath(os.path.join(download_folder))
        rl_folder = os.path.normpath(os.path.join(destination_folder))
        ds_folder = os.path.normpath(os.path.join(rl_folder, dataset))

        print("Relocating dataset", dataset)

        match dataset:
            case "yahoo_s5" | "ghl" | "nasa_msl_smap":
                if dataset == "yahoo_s5":
                    ds_zipfile = os.path.normpath(os.path.join(dw_folder, dataset + ".tgz"))
                else:
                    ds_zipfile = os.path.normpath(os.path.join(dw_folder, dataset + ".zip"))

                new_folder = extract_and_get_new_filename(ds_zipfile, rl_folder, dataset)
                os.rename(new_folder, ds_folder)
                print()

            case "mgab" | "nab":
                ds_zipfile = os.path.normpath(os.path.join(dw_folder, dataset + ".zip"))
                new_folder = extract_and_get_new_filename(ds_zipfile, rl_folder, dataset)
                os.rename(new_folder, ds_folder)

                contents = os.listdir(ds_folder)
                for dir_file in contents:
                    if dir_file not in _MAINTAIN_FILE_DIR[dataset]:
                        if os.path.isdir(os.path.join(ds_folder, dir_file)):
                            shutil.rmtree(os.path.join(ds_folder, dir_file))
                        else:
                            os.remove(os.path.join(ds_folder, dir_file))
                print()

            case "smd" | "ucr" | "exathlon":
                ds_zipfile = os.path.normpath(os.path.join(dw_folder, dataset + ".zip"))
                new_folder = extract_and_get_new_filename(ds_zipfile, rl_folder, dataset)

                if dataset == "smd":
                    dataset_folder = os.path.normpath(os.path.join(str(new_folder),
                                                                   "ServerMachineDataset"))
                    final_folder = os.path.normpath(os.path.join(rl_folder,
                                                                 "ServerMachineDataset"))
                elif dataset == "ucr":
                    dataset_folder = os.path.normpath(os.path.join(str(new_folder),
                                                                   "UCR_TimeSeriesAnomalyDatasets2021",
                                                                   "FilesAreInHere",
                                                                   "UCR_Anomaly_FullData"))
                    final_folder = os.path.normpath(os.path.join(rl_folder,
                                                                 "UCR_Anomaly_FullData"))
                else:
                    dataset_folder = os.path.normpath(os.path.join(str(new_folder),
                                                                   "data",
                                                                   "raw"))
                    final_folder = os.path.normpath(os.path.join(rl_folder,
                                                                 "raw"))

                shutil.move(dataset_folder, rl_folder)
                os.rename(final_folder, ds_folder)
                shutil.rmtree(new_folder)
                print()

            case "kitsune":
                ds_zipfile = os.path.normpath(os.path.join(dw_folder, dataset + ".zip"))
                new_folder = extract_and_get_new_filename(ds_zipfile, rl_folder, dataset)
                os.rename(new_folder, ds_folder)
                print()


def process_datasets(to_process: str | Iterable[str] = "all",
                     datasets_folder: str | bytes | os.PathLike = _DEFAULT_FOLDER) -> None:
    """Process all the specified extracted datasets.

    Parameters
    ----------
    to_process : str or Iterable[str], default="all"
        The datasets that the user wants to be processed by organizing the
        folders or by extracting sub-folders. The aim is to build the data
        folder as it has been used while building this tool.

    datasets_folder : str | bytes | os.PathLike, default="./data"
        The folder in which there are the relocated datasets.

    Returns
    -------
    None
    """
    create_path_if_not_exist(datasets_folder,
                             "Successfully accessed the datasets' folder")

    if to_process == "all":
        to_process = set(_PUBLIC_DATASETS).union(_AVAILABLE_DATASETS)

    for dataset in to_process:
        ds_folder = os.path.normpath(os.path.join(datasets_folder, dataset))

        print("Processing the relocated datasets")

        match dataset:
            case "kitsune":
                folder_to_visit = [ds_folder]
                while len(folder_to_visit) != 0:
                    folder = folder_to_visit.pop()
                    new_contents = os.listdir(folder)

                    for dir_file in new_contents:
                        if os.path.isdir(os.path.join(folder, dir_file)):
                            folder_to_visit.append(os.path.join(folder, dir_file))
                        else:
                            archive = os.path.normpath(os.path.join(folder, dir_file))
                            if dir_file.endswith(".gz"):
                                new_file = extract_and_get_new_filename(archive, folder, dir_file, is_archive=False)
                                os.remove(archive)
                print()

            case "exathlon":
                folder_to_visit = [ds_folder]
                while len(folder_to_visit) != 0:
                    folder = folder_to_visit.pop()
                    new_contents: list[str] = os.listdir(folder)
                    has_zetas = any([".z0" in e for e in new_contents])

                    if has_zetas:
                        script_path = os.path.normpath(os.path.join(ds_folder, "finish_process.sh"))
                        with open(script_path, "w") as f:
                            f.write("#!/bin/bash\n\n"
                                    "for (( i=1; i<=10; i++ )); do\n"
                                    "  DIR=\"./app$i\"\n"
                                    "  for subdir in $(find $DIR -mindepth 1 -type d); do\n"
                                    "    ZIP_FILE_PATH=$(find $subdir -iname \\*.zip)\n"
                                    "    ZIP_FILE_NAME=${ZIP_FILE_PATH##*/}\n"
                                    "    # zip -s0 caused some issues with some files so we used cat instead\n"
                                    "    ZIP_FILE_PATH_NO_EXT=${ZIP_FILE_PATH%.*}\n"
                                    "    echo $ZIP_FILE_PATH\n"
                                    "    echo $ZIP_FILE_NAME\n"
                                    "    echo $ZIP_FILE_PATH_NO_EXT\n"
                                    "    cat $ZIP_FILE_PATH_NO_EXT.z0* $ZIP_FILE_PATH > $DIR/$ZIP_FILE_NAME\n"
                                    "    unzip $DIR/$ZIP_FILE_NAME -d $DIR\n"
                                    "    rm $DIR/$ZIP_FILE_NAME\n"
                                    "    rm -rf $subdir\n"
                                    "  done\n"
                                    "done\n")

                        print("Split-archives are not supported by the python "
                              "community. We are deeply sorry to inform you "
                              f"that you must execute the script {script_path} "
                              f"to finish to process the {dataset}")
                        # zetas = [e for e in new_contents if ".z0" in e]
                        # zetas = sorted(zetas)
                        # zip_file = list(set(new_contents).difference(zetas))[0]
                        # temp = zip_file.split(".")[0] + "_temp.zip"
                        # zip_path = os.path.normpath(os.path.join(folder, zip_file))
                        # temp_path = os.path.normpath(os.path.join(folder, temp))
                        #
                        # for zeta_file in zetas:
                        #     zeta_path = os.path.normpath(os.path.join(folder, zeta_file))
                        #     with open(temp_path, "ab") as f, open(zeta_path, "rb") as f_1:
                        #         f.write(f_1.read())
                        #
                        # with open(temp_path, "ab") as f, open(zip_path, "rb") as f_1:
                        #     f.write(f_1.read())
                        #
                        # shutil.unpack_archive(temp_path, Path(temp_path).parent)
                        # shutil.rmtree(folder)
                    else:
                        for dir_file in new_contents:
                            if os.path.isdir(os.path.join(folder, dir_file)):
                                folder_to_visit.append(os.path.join(folder, dir_file))
                            else:
                                archive = os.path.normpath(os.path.join(folder, dir_file))
                                new_file = extract_and_get_new_filename(archive, folder, dir_file)
                                os.remove(archive)

                            if dir_file == "ground_truth.zip":
                                shutil.move(os.path.join(folder, "data", "raw", "ground_truth.csv"), folder)
                                shutil.rmtree(os.path.join(folder, "data"))


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
    parser = argparse.ArgumentParser(prog="setup-dataset.py",
                                     description="Downloads and setup all the "
                                                 "used datasets from the repo")
    parser.add_argument("--no-download",
                        action="store_const",
                        const=True,
                        default=False,
                        help="does not execute datasets download")
    parser.add_argument("--no-relocate",
                        action="store_const",
                        const=True,
                        default=False,
                        help="does not execute datasets relocation")
    parser.add_argument("--no-process",
                        action="store_const",
                        const=True,
                        default=False,
                        help="does not execute datasets process")
    parser.add_argument("-d", "--dataset",
                        action="extend",
                        nargs="+",
                        help="defines which datasets must be used for the "
                             "download, relocation or process operations. You "
                             "must specify at least one element. If it is left "
                             "empty, all datasets will be used",
                        choices=set(_PUBLIC_DATASETS).union(_AVAILABLE_DATASETS))
    parser.add_argument("-f", "--folder",
                        help="specifies the location of the folder where the "
                             "user wants to download, relocate and process the "
                             "datasets. Thus, this is the final folder that the"
                             " user will se after the whole process will have "
                             "finished. If this value is passed, the download "
                             "folder option and relocation folder option will "
                             "be ignored")
    parser.add_argument("--download-folder",
                        help="specifies the folder in which the datasets must "
                             "be downloaded. If specified, the relocation "
                             "folder must be given, otherwise it will be the "
                             "same as the download folder")
    parser.add_argument("--relocate-folder",
                        help="specifies the folder in which the datasets must "
                             "be relocated. If specified, the download folder "
                             "must be given, otherwise, it will be the same as "
                             "the relocation folder")

    output = parser.parse_args(argv[1:])
    arguments = vars(output)

    if arguments["folder"] is not None and not os.path.exists(arguments["folder"]):
        parser.error("The folder must be a valid path")
    if arguments["download_folder"] is not None and not os.path.exists(arguments["download_folder"]):
        parser.error("The download_folder must be a valid path")
    if arguments["relocate_folder"] is not None and not os.path.exists(arguments["relocate_folder"]):
        parser.error("The relocate_folder must be a valid path")

    return arguments


if __name__ == "__main__":
    options = process_arguments(sys.argv)

    if options["dataset"] is not None:
        _download_datasets = set(options["dataset"]).difference(_AVAILABLE_DATASETS)
        _selected_datasets = set(options["dataset"])
    else:
        _download_datasets = "all"
        _selected_datasets = "all"

    if options["folder"] is not None:
        _download_folder = options["folder"]
        _relocate_folder = options["folder"]
    else:
        if options["download_folder"] is not None:
            if options["relocate_folder"] is None:
                _download_folder = options["download_folder"]
                _relocate_folder = options["download_folder"]
            else:
                _download_folder = options["download_folder"]
                _relocate_folder = options["relocate_folder"]
        elif options["relocate_folder"] is not None:
            _download_folder = options["relocate_folder"]
            _relocate_folder = options["relocate_folder"]
        else:
            _download_folder = _DEFAULT_FOLDER
            _relocate_folder = _DEFAULT_FOLDER

    if not options["no_download"]:
        download_datasets(_download_datasets, _download_folder)

    if not options["no_relocate"]:
        relocate_datasets(_selected_datasets, _download_folder, _relocate_folder)

    if not options["no_process"]:
        process_datasets(_selected_datasets, _relocate_folder)
