import napari

from util import load_tif_as_zarr, extract_region_instance_id_arrays

image_path = "data/philips/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16.tiff"
label_path = "data/philips/cd8/RE-000072-1_1_4-CD8_CRC_CancerScout-2024-07-12T11-04-16_annotations.json"

regions = extract_region_instance_id_arrays(label_path)

scale_level = 0
data = load_tif_as_zarr(image_path, scale_level)

for bb, seg in regions.values():
    image = data[bb]
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(seg)
    napari.run()
