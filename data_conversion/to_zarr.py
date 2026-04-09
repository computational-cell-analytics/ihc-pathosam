# Note: bfconvert creates broken tifs (great ...)
# so we have to convert to zarr to process some of the files correctly.
import multiprocessing as mp
import os
import tempfile
from concurrent import futures
from glob import glob
from subprocess import run

import tifffile
import zarr
from elf.wrapper.resized_volume import ResizedVolume
from nifty.tools import blocking
from tqdm import tqdm


def convert_to_zarr(input_path, output_path, n_scales=6):
    # First we need to convert to a plain tif with bfcovert (nothing else can read this f'd up file anymore).
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, "image.tif")
        run(["bfconvert", input_path, tmp_path])

        # Then read the data with imageio and transpose the chunkdim.
        input_ = tifffile.TiffFile(tmp_path).pages[0].asarray().transpose((1, 2, 0))

    chunk_shape = (512, 512, 3)
    shard_shape = (4 * 512, 4 * 512, 3)

    output = zarr.open(output_path, mode="a")
    n_threads = 4

    def copy(inp, out):
        tiles = blocking([0, 0], out.shape[:2], shard_shape[:2])
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
        out_shape = tuple(sh // 2 for sh in shape[:2]) + (3,)
        print("Rescaling from", shape, "to", out_shape)
        input_scale = ResizedVolume(input_scale, shape=out_shape)
        output_scale = output.create_array(
            f"s{scale}/image", shape=input_scale.shape, chunks=chunk_shape, shards=shard_shape, dtype=input_.dtype,
        )
        copy(input_scale, output_scale)
        input_scale = output_scale


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        inputs = sorted(glob(os.path.join(args.input, "*.tif")))
        output_folder = args.output
        os.makedirs(output_folder, exist_ok=True)
        for inp in inputs:
            fname = os.path.basename(inp)
            out = os.path.join(output_folder, fname.replace("ome.tif", "zarr"))
            convert_to_zarr(inp, out)

    else:
        convert_to_zarr(args.input, args.output)


if __name__ == "__main__":
    main()
