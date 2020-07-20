"""
Calculate components at a given similarity threshold for repeats
"""


from pathlib import Path

import click
import numpy as np
from cherno1.components import find_components


@click.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.argument("threshold", type=click.INT)
@click.option(
    "--outdir",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Directory to save the output.",
)
@click.option(
    "--n", default=None, type=click.INT, help="Only process the first N repeats."
)
def components(file, threshold, outdir, n):
    """Calculate components at similarity >= THRESHOLD from repeats file FILE."""
    outfile = Path(outdir) / f"components_{threshold:02}.npy"
    data = np.load(file)[:n]
    components = find_components(data, threshold)
    np.save(outfile, components)


if __name__ == "__main__":
    components()
