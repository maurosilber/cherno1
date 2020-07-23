from functools import partial

import click
import numpy as np
import tqdm
from cherno1.common import castdown
from cherno1.graph import build_igraph
from p_tqdm import p_umap


def func(sn_repeats, sn, min_similarity, progress=False):
    degree = np.zeros((len(sn_repeats), 34), dtype=int)
    partitions = np.zeros((len(sn_repeats), 34), dtype=int)
    for s in tqdm.trange(min_similarity, 33, disable=not progress):
        graph = build_igraph(sn_repeats, s)
        degree[:, s] = graph.degree()
        partitions[:, s] = graph.community_infomap().membership

    np.save(f"degree_{sn}.npy", castdown(degree))
    np.save(f"partitions_{sn}.npy", castdown(partitions))


@click.command()
@click.argument("repeat_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("supernode_file", type=click.Path(exists=True, dir_okay=False))
def cli(repeat_file, supernode_file):
    repeats = np.load(repeat_file)["repeats"]
    supernodes = np.load(supernode_file)["25"]
    sns = np.unique(supernodes)
    sns_repeats = (repeats[supernodes == sn] for sn in sns)
    p_umap(partial(func, min_similarity=10), sns_repeats, sns)


if __name__ == "__main__":
    cli()
