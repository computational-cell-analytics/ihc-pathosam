import h5py
import zarr


def zarr_to_h5(input_path, input_key, output_path, output_key):
    print("Loading data ...")
    data = zarr.open(input_path, mode="r")[input_key][:]

    print("Writing data ...")
    with h5py.File(output_path, mode="a") as f:
        f.create_dataset(output_key, data=data, compression="gzip")


# TODO use arg-parse
def main():
    input_path = "/mnt/lustre-grete/tmp/u12086/tmpma1dpfg3/tmp_seg.zarr"
    input_key = "seg"

    output_path = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/ihc-pathosam/data/predictions-wsi/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16.h5"  # noqa
    output_key = "segmentation/patho-sam"

    zarr_to_h5(input_path, input_key, output_path, output_key)


if __name__ == "__main__":
    main()
