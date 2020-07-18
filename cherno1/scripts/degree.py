from pathlib import Path

import click
import numpy as np
from cherno1.degree import degree_by_similarity


@click.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--outdir",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Directory to save the output.",
)
@click.option(
    "--progress",
    type=click.INT,
    default=0,
    help="To see a progress bar, pass a batch size number.",
)
@click.option(
    "--n", default=-1, type=click.INT, help="Only process the first N repeats."
)
def degrees(file, outdir, n, progress):
    """Calculate components at similarity >= THRESHOLD from repeats file FILE."""
    outfile = Path(outdir) / "degree_by_similarity.npy"
    data = np.load(file)[:n]
    degrees = degree_by_similarity(data, progress=progress, batch_size=progress)
    np.save(outfile, degrees)


if __name__ == "__main__":
    degrees()
