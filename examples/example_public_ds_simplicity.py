import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rc("font", family="serif", serif=["Computer Modern Roman"], size=12)
matplotlib.rc("text", usetex=True)

datasets = ["Yahoo", "UCR", "SMD", "NAB", "MGAB", "GHL", "NASA"]
simplicity_out = pd.read_csv("../output/datasets_simplicity/simplicity_scores.csv")

for column in ["Constant score", "Moving average score", "Moving standard deviation score", "Mixed score"]:
    mixed_scores = dict()
    simplicity_df = simplicity_out.drop(columns=set(simplicity_out.columns).difference(["Dataset", column])).groupby("Dataset")
    
    for ds in datasets:
        frame = simplicity_df.get_group(ds)
        mixed_scores[ds] = frame[column].values
    
    nans = {key: np.sum(np.isnan(mixed_scores[key])) for key in mixed_scores}
    values = [mixed_scores[key][~np.isnan(mixed_scores[key])] for key in mixed_scores]
    names = [key for key in mixed_scores]
    
    f, ax = plt.subplots(figsize=(5, 4))
    sns.violinplot(values, scale="width", cut=0, ax=ax)
    ax.set_xticks(range(len(values)), datasets)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(column)
    ax.set_title(f"Violin+Box plot of {column.lower()}")
    plt.show()
