# Note: bfconvert creates broken tifs (great ...)
# so we have to convert to zarr to process some of the files correctly.
import sys
import multoprocessing as mp
from concurrent import futures

import zarr
from elf.wrapper import ResizedVolume
from nifty.tools import blocking
from tqdm import tqdm

sys.path.append("..")
from util import load_tif_as_zarr  # noqa


def convert_to_zarr(input_path, output_path, n_scales=6):
    input_ = load_tif_as_zarr(input_path)
    # TODO handle channels and transpose?
    breakpoint()

    chunk_shape = (1024, 1024)
    shard_shape = (6 * 1024, 6 * 1024)

    output = zarr.open(output_path, mode="a")
    n_threads = mp.cpu_count()

    def copy(inp, out):
        tiles = blocking([0, 0], out.shape, shard_shape)
        n_tiles = tiles.numberOfBlocks

        def copy_shard(tile_id):
            tile = tiles.getBlock(tile_id)
            bb = tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end))
            data = inp[bb]
            out[bb] = data

        with futures.ThreadPoolExecutor(n_threads) as tp:
            list(tqdm(
                tp.map(copy_shard, range(n_tiles)), total=n_tiles
            ))

    # Copy scale 0.
    s0 = output.create_array(
        "s0/image", shape=input_.shape, chunks=chunk_shape, shards=shard_shape, dtype=input_.dtype
    )
    copy(input_, s0)

    # Downscale.
    input_scale = s0
    for scale in range(1, n_scales):
        shape = input_scale.shape
        out_shape = tuple(sh // 2 for sh in shape)
        print("Rescaling from", shape, "to", out_shape)
        input_scale = ResizedVolume(input_scale, shape=out_shape)
        output_scale = output.create_array(
            f"s{scale}/image", shape=input_scale.shape, chunks=chunk_shape, shards=shard_shape, dtype=input_.dtype,
        )
        copy(input_scale, output_scale)
        input_scale = output_scale
