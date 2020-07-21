"""
Calculate components at a given similarity threshold for repeats
"""
from pathlib import Path

import click
import numpy as np
from cherno1.project import project


@click.command()
@click.argument("array_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("repeat_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--outdir",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Directory to save the output.",
)
def projection(array_file, repeat_file, outdir):
    filename = Path(repeat_file).name.replace("components", "supernodes")
    outfile = Path(outdir) / filename

    arrays = np.load(array_file)
    repeats = np.load(repeat_file)
    supernodes = project(repeats, arrays)
    np.save(outfile, supernodes)


if __name__ == "__main__":
    projection()
