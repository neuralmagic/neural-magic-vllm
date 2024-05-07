import torch
import numpy
import pickle as pkl
from dataclasses import dataclass
import math

from typing import List
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

@dataclass
class Data:
    m: int
    k: int
    n: int
    description: str
    time: float
    tflops: float

def plot_graph(Xs : List[List[str]],
               Ys : List[List[str]],
               labels : List[str],
               title: str,
               save_filename: str):

    plt.set_cmap('Dark2')

    fig, ax = plt.subplots()
    ax.set_title(title) 
    ax.set_ylabel("tflops")
    ax.set_xlabel("gemm problem")

    ax.locator_params(axis='x', nbins=25)

    for xs, ys, label in zip(Xs, Ys, labels):
        ax.scatter(xs, ys, label=label, s=1.0)

    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::int(math.ceil(len(xticks) / 25))]) # set new tick positions
    ax.tick_params(axis='x', rotation=90) # set tick rotation
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_filename)
    plt.show()
    plt.close(fig)

def parse_mkn(mkn_str: str):
    # mkn_str : MKN=(16x1024x512)
    mkn_tuple = mkn_str.split("=")[1]
    # mkn_tuple : (16x1024x512)
    mkn_prod = mkn_tuple[1:-1]
    # mkn_prod : 16x1024x512
    mkn_tuple = tuple(mkn_prod.split("x"))
    return (int(mkn_tuple[0]), int(mkn_tuple[1]), int(mkn_tuple[2]))

def comparison_to_data(comparison):
    m, k, n = parse_mkn(comparison.sub_label)
    t_ops = 2*m*k*n / 1024 / 1024 / 1024 / 1024
    tflops = t_ops / comparison.median 
    return Data(m, k, n, comparison.task_spec.description, comparison.median, tflops)

def plot_data(data, title, save_filename):

    # group data by description
    groups = {}
    for d in data:
        if groups.get(d.description) is None:
            groups[d.description] = []
        groups[d.description].append(d)

    Xs = []
    Ys = []
    labels = []
    for label, values in groups.items():
        # get all x and y values
        Xs.append(list(map(lambda v: f"MKN={v.m}x{v.k}x{v.n}", values)))
        Ys.append(list(map(lambda v: v.tflops, values)))
        labels.append(label)

    plot_graph(Xs, Ys, labels, title, save_filename)

def plot_measurements(measurements, title, save_filename):
    data = list(map(lambda x: comparison_to_data(x), measurements))
    plot_data(data, title, save_filename)

def main():

    # load the comparisons
    comparisons = None
    with open("./comparisons.pkl", "rb") as f:
        comparisons = pkl.load(f)

    # make data
    data = list(map(lambda x: comparison_to_data(x), comparisons))

    # plot data
    plot_data(data)

if __name__ == '__main__':
    main()
