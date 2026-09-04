# QuPath annotation pyramid export

`export_hdf5.py` converts the QuPath annotations for each VSI sample into a
label-only HDF5 pyramid. It does not store any image or VSI thumbnail data.

The exporter reads the WSI pyramid dimensions and downsamples from the image
server metadata serialized in each `.qpdata` file. This metadata is required
because the provided VSI files do not include their external `.ets` tile files.

The default labels are:

| Value | Class |
| ---: | --- |
| 0 | Background |
| 1 | Normal pancreas |
| 2 | Tumor |
| 255 | Unclassified or review required |

TM11 contains one unclassified annotation. The exporter keeps it as label 255.

## Export

Install the Python dependencies and QuPath. Ensure that the `qupath` executable
is available on `PATH`. Then run:

```bash
python export_hdf5.py data --output-dir hdf5 --overwrite
```

Use `--qupath /path/to/QuPath` if QuPath is not on `PATH`. Run
`python export_hdf5.py --help` for class mapping, worker, and chunk-size options.

To validate against converted Zarr WSIs and make the label levels match the
exact `s*/image` array shapes, pass the directory containing the matching
`SAMPLE.zarr` stores:

```bash
python export_hdf5.py data \
    --zarr-dir /path/to/converted_zarr_he \
    --output-dir hdf5 \
    --overwrite
```

The exporter checks that each Zarr `s0/image` shape matches the dimensions
serialized with the QuPath annotations and that every annotation is within the
native image bounds. It then writes one HDF5 label level for every Zarr image
level using the same height and width.

Each HDF5 file contains a `labels` group. Its datasets are named by pyramid
level:

```text
labels/0   full-resolution labels
labels/1   2× downsampled labels
labels/2   4× downsampled labels
...
```

The level dimensions and downsamples match the WSI pyramid metadata. The exporter
uses `bioimage_py` to process the labels block-wise and out of core. Level 0 is
rasterized in blocks, so the full gigapixel mask is never held in memory. Each
subsequent level uses a nearest-neighbor 2× stride from the immediately preceding
level, cropped to its exact target dimensions. It never resamples directly from
level 0. The default is one worker with 512 × 512-pixel blocks to keep peak
memory low.

The datasets use chunked LZF compression and an implicit background fill value.
Root and dataset attributes contain the class mapping, pixel sizes, objective
magnification, pyramid dimensions, and source information.

## Check in napari

Open one pyramid level from all exported samples:

```bash
python view_hdf5_napari.py hdf5 --level 7
```

Level 7 is the common 128× downsampled level. Use `--level 0` for the native 20×
annotation grid. The viewer uses lazy Dask arrays, so it does not load the full
level into memory.

The default `--level -1` selects the coarsest level in each file. TM105 has
levels 0 through 7. The other samples have levels 0 through 8.

## Portable image and label files

`export_combined_hdf5.py` combines one Zarr image level with the corresponding
annotation level. By default it exports level 4 and writes one file per matched
sample to `combined_hdf5_level4`. Each output contains `raw` (YXC RGB image) and
`labels` (YX annotation mask) datasets:

```bash
python export_combined_hdf5.py
```

The input directories, output directory, level, dataset keys, compression, and
sample selection can all be overridden; run
`python export_combined_hdf5.py --help` for details.

## Resolution and magnification

The VSI metadata reports a 20× acquisition objective and a base sampling of about
0.274 µm/pixel for all six slides. Pyramid level `n` has a downsample of `2**n`
and a pixel size of approximately `0.274 * 2**n` µm.
