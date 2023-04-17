# Anomaly detection in time series for energy efficiency
This is the repository containing all the codes related to my Master of Science thesis in Computer Science and Engineering at Politecnico di Milano. The description of the thesis and of the repository structure and code structure is postponed to next sections to keep the first sections short. The first components I will describe are the ones needed to reproduce the experiments and to download and set up the datasets using automated scripts. I don't want you to lose time, most of the things just involve just running a script.

# Where to find anomalearn updated code?
The development of anomalearn has been moved to a dedicated repository to be correctly synced with PyPI for versioning. The link of the anomalearn repository is: https://github.com/marcopetri98/anomalearn. If the repository is still private, it means that you visited this repository too close to the graduation day. It will be public soon.

# How to reproduce the experiments
In my thesis I use some external libraries, public datasets and state-of-the-art methods. If you want to do one of the following:

* Use the code of the repository
* Download all the datasets used in this thesis
* Download all the methods to which this thesis is compared
* Reproduce the results of the thesis by running scripts

This section explains everything you need from downloading the required packages to execute the scripts (including the scripts used to download datasets and methods).

## Get the library
In my thesis I used several packages. To get all the needed packages to install `anomalearn`, make sure you use python 3.10 and create the virtual environment using:

```
conda create -n <env-name> python=3.10
conda activate <env-name>
pip install -e .
```

The reason why the installation uses the editable option `-e` is that the library is still in development at the moment, and there isn't a stable release. The API are subject to abrupt change, and it is not still published on PyPI. Sooner the installation will become available through PyPI as any other library by running `pip install anomalearn`. However, for the moment this is the workaround to be used to install the library. The advantage is that with the editable option you can git clone the repository and pull the latest updates without needing to reinstall the library if the requirements did not change.

If either you don't have conda or you don't know what conda is, I invite you to follow this link: https://docs.conda.io/en/latest/miniconda.html

## Run the tests
Every library must be tested before being distributed. Especially, if the library is published on PyPI it should have automated testing providing a good coverage level. Assume you have downloaded the repository at `/usr/yourname/repositories/anomalearn`, you need to follow these steps to test the library:

```
cd /usr/yourname/repositories/anomalearn
conda activate <env/name>
python -m unittest discover
```

## Get the datasets
In my thesis I use several public datasets. To increase reproducibility, you can find the script ``setup_datasets.py``. The script accepts some arguments. By default the script downloads, extracts and rename folders. The usage with all default settings is:

```
python ./setup_datasets.py
```

**Be aware that the script does not download datasets that are not directly downloadable**: for those datasets (such as Yahoo! Webscope S5) you need to request access to them, download them and to positionate them in the folder in which you downloaded the other datasets. Then, you need also to rename them with the expected name by the program, here is the list:

* Yahoo! Webscope S5: the filename must be "yahoo_s5.tgz".

If you want to download some of the datasets, you can specify that by using the appropriate optional argument:

```
python ./setup_datasets.py -d ucr smd nab mgab
```

Will download, extract and rename the datasets: ucr, smd, nab and mgab.

The script is a program as any other linux program. You can call the man page with the following (if you want to know about the optional parameters and their usage):

```
python ./setup_datasets.py -h
```

## Get the repositories for comparison
In my thesis I compare my methods with several state-of-the-art methods. Many of those methods have a public repository available, from which we can download their code of their model. The script ``setup_repos.py`` is a utility similar to ``setup_datasets.py``. It can be used to download all the repos in their folder. The default usage is:

```
python ./setup_repos.py
```

The methods can also be downloaded to a different location with respect to the standard one. However, scripts are thought to work with default locations. If you change it, you may need to adjust paths in scripts you are going to execute.

This script is a program as any other linux program. You can call the man page with the following (if you want to know about the optional parameters and their usage):

```
python ./setup_repos.py -h
```

# Datasets description
In my thesis I used many datasets to evaluate the methods I developed and state-of-the-art methods against each other. The datasets I used are: Yahoo Webscope S5, UCR, SMD, NASA-MSL, NASA-SMAP, NAB, MGAB, Kitsune, GHL and Exathlon. Here you can find a table with the datasets, reference to the paper presenting them and the link to the dataset.

| Dataset | Paper | Repository |
| ------- | ----- | ---------- |
| Exathlon | https://doi.org/10.14778/3476249.3476307 | https://github.com/exathlonbenchmark/exathlon |
| GHL | https://doi.org/10.48550/arXiv.1612.06676 | https://kas.pr/ics-research/dataset_ghl_1 |
| Kitsune | http://dx.doi.org/10.14722/ndss.2018.23204 | https://github.com/ymirsky/KitNET-py |
| MGAB | None | https://doi.org/10.5281/zenodo.3760086 |
| NAB | https://doi.org/10.1016/j.neucom.2017.04.070 | https://github.com/htm-community/NAB |
| NASA-SMAP | https://doi.org/10.1145/3219819.3219845 | https://github.com/khundman/telemanom |
| NASA-MSL | https://doi.org/10.1145/3219819.3219845 | https://github.com/khundman/telemanom |
| SMD | https://doi.org/10.1145/3292500.3330672 | https://github.com/smallcowbaby/OmniAnomaly |
| UCR | https://doi.org/10.1109/TKDE.2021.3112126 | https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ |
| Yahoo! Webscope S5 | None | https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70 |

These datasets are all public datasets. However, not all of them are directly downloadable. For instance, Yahoo! Webscope S5 can be downloaded upon request approval made to Yahoo via their website. Therefore, I will speak about public datasets (available for download from a link) and about accessible datasets (downloadable after request acceptance).