import torch
import numpy
import pickle as pkl
from dataclasses import dataclass
import math
import time

from typing import List
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections.abc import Iterable

SAMPLE_XTICKS=True

@dataclass
class Data:
    m: int
    k: int
    n: int
    description: str
    time: float
    tflops: float

def plot_subgraph(ax,
                  Xs : List[List[str]],
                  Ys : List[List[str]],
                  labels : List[str],
                  title: str):

    ax.set_title(title) 
    ax.set_ylabel("tflops")
    ax.set_xlabel("gemm problem")

    ax.locator_params(axis='x', nbins=25)

    for xs, ys, label in zip(Xs, Ys, labels):
        ax.scatter(xs, ys, label=label, s=1.0)

    xticks = ax.get_xticks()
    if SAMPLE_XTICKS:
        ax.set_xticks(xticks[::int(math.ceil(len(xticks) / 25))]) # set new tick positions
    ax.tick_params(axis='x', rotation=90) # set tick rotation
    return

# 
# Xs : List[List[List[str]]]
#        |    |     |----> List of data (numbers really)
#        |    |----------> List of implementations. ([pytorch_impl, cutlass_impl])             
#        |---------------> List of subplots (Different models maybe) 
def plot_graph(Xs : List[List[List[str]]],
               Ys : List[List[List[str]]],
               labels : List[List[str]],
               title: List[str],
               save_filename: str):

    plt.set_cmap('Dark2')

    n_subplots = len(Xs)
    fig, axs = plt.subplots(ncols = 1, nrows = n_subplots) 

    if not isinstance(axs, Iterable):
        axs = [axs]
    for idx, ax in enumerate(axs):
        t = f'{title[idx]} \n {labels[idx][0]} vs \n   {labels[idx][1]}'  
        l = list(map(lambda x: x.split('_')[0], labels[idx]))
        plot_subgraph(ax, Xs[idx], Ys[idx], l, t)

    plt.legend()
    plt.grid(axis='x', alpha=0.2, color='gray', linestyle='-')
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

def plot_data(all_data, titles, save_filename):

    data_Xs = []
    data_Ys = []
    data_labels = []

    for data in all_data: 
        # group data by description. i.e by pytorch_impl vs cutlass_impl
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
        data_Xs.append(Xs)
        data_Ys.append(Ys)
        data_labels.append(labels)

    plot_graph(data_Xs, data_Ys, data_labels, titles, save_filename)

def plot_measurements(measurements, title, save_filename):
    data = list(map(lambda x: comparison_to_data(x), measurements))
    plot_data([data], [title], save_filename)

def plot_model_measurements(measurements, save_prefix, model_name=None):

    data = list(map(lambda x: comparison_to_data(x), measurements))

    from weight_shapes import WEIGHT_SHAPES
    models = [model_name] if model_name is not None else list(WEIGHT_SHAPES.keys())[1:]

    def get_kns(model):
        KNs = []
        for layer in WEIGHT_SHAPES[model]:
            KNs.append((layer[0], layer[1]))
        return KNs

    def group_data(data):
        # Group by KNs
        groups = {}
        for d in data:
            kn = (d.k, d.n)
            if groups.get(kn) is None:
                groups[kn] = []
            groups[kn].append(d)

        # Sort each group by M
        for key in groups.keys():
            groups[key].sort(key = lambda d: d.m)

        KNs = list(groups.keys())
        KNs.sort(key = lambda kn: kn[0])
        grouped_data = []
        for kn in KNs:
            grouped_data.extend(groups[kn])
        return grouped_data

    plot_models = []
    for model in models:
        KNs = get_kns(model)
        # If the measurements have all the KNs - then plot it
        have_all_layer_data = all(
                                 map(lambda kn:
                                     len(list(filter(lambda d: d.k == kn[0] and d.n == kn[1], data))) != 0, KNs))
        if have_all_layer_data:
            plot_models.append(model)

    plot_model_data = []
    plot_model_data_titles = []
    for plot_model in plot_models:
        KNs = get_kns(plot_model)
        model_data = list(filter(lambda d: (d.k, d.n) in KNs, data))
        # Sort data by KNs and then by M
        model_data = group_data(model_data)
        plot_model_data.append(model_data)
        plot_model_data_titles.append(plot_model)

    global SAMPLE_XTICKS
    sample_xticks_old = SAMPLE_XTICKS
    SAMPLE_XTICKS=False
    for plt_data, plt_title in zip(plot_model_data, plot_model_data_titles):
        plot_data([plt_data], [plt_title], f'{save_prefix}-{plt_title.replace("/", "_").replace(".", "_")}.png')
    SAMPLE_XTICKS = sample_xticks_old

def main(args):

    # load the comparisons
    measurements = None
    with open(f"{args.input_pkl}", "rb") as f:
        measurements = pkl.load(f)

    if args.model_bench:
        timestamp = int(time.time())
        save_filename = f"model_bench-{timestamp}.png"
        plot_model_measurements(measurements, save_filename)
    else:
        plot_measurements(measurements, "gemm+dequant", "gemm_dequant.png")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="plot bench measurements pkl")
    parser.add_argument("--input-pkl", "-i", required=True, type=str)
    parser.add_argument("--model-bench", action="store_true")
    args = parser.parse_args()
    main(args)
