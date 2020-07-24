from pathlib import Path

import click
import numpy as np
from cherno1.common import castdown
from cherno1.graph import build_igraph
from p_tqdm import p_umap


def func(supernode_repeats, supernode, similarity):
    p = Path(f"degree_{supernode}_{similarity}.npy")
    if p.exists():
        return

    graph = build_igraph(supernode_repeats, similarity)
    degree = graph.degree()
    partitions = graph.community_infomap().membership
    np.save(p, castdown(degree))
    np.save(f"partitions_{supernode}_{similarity}.npy", castdown(partitions))


@click.command()
@click.argument("repeat_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("supernode_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("supernodes", nargs=-1, type=click.INT)
@click.option(
    "--min-similarity", type=click.INT, default=10, help="Upto this similarity.",
)
def cli(repeat_file, supernode_file, supernodes, min_similarity):
    repeats = np.load(repeat_file)["repeats"]
    supernode_array = np.load(supernode_file)["25"]

    if not supernodes:
        supernodes = np.unique(supernode_array)

    def _data(supernodes, min_similarity):
        for s in range(33, min_similarity, -1):
            for sn in supernodes:
                yield repeats[supernode_array == sn], sn, s

    p_umap(
        lambda x, y: func(*x),
        _data(supernodes, min_similarity),
        range(len(supernodes) * (33 - min_similarity)),
    )


if __name__ == "__main__":
    cli()
