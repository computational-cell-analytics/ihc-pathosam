# Note: bfconvert creates broken tifs (great ...)
# so we have to convert to zarr to process some of the files correctly.
import sys
import zarr
from concurrent import futures

from nifty.tools import blocking
from elf.wrapper import ResizedVolume

sys.path.append("..")
from util import load_tif_as_zarr  # noqa


def convert_to_zarr(input_path, output_path, n_scales=6):
    input_ = load_tif_as_zarr(input_path)

    chunk_shape = (1024, 1024)
    shard_shape = (6 * 1024, 6 * 1024)

    output = zarr.open(output_path, mode="a")
