"""Export a tissue classifier prediction as QuPath annotations (GeoJSON).

The annotations follow the convention of the PDAC QuPath projects: one polygon annotation per connected region,
classified as "Normal pancreas" or "Tumor" (plus "Other tissue"), in full resolution pixel coordinates of the slide.
Background (glass) is not exported. Import the file in QuPath with File > Import objects from file, or by dragging
it onto the opened slide.
"""
import json
import uuid
import argparse

import cv2
import h5py
import numpy as np

from tissue_util import get_pyramid


# QuPath class names and colors (the colors of the prediction figures).
QUPATH_CLASSES = {
    1: ("Normal pancreas", [42, 120, 214]),
    2: ("Tumor", [74, 58, 167]),
    3: ("Other tissue", [138, 109, 75]),
}


def mask_to_polygons(mask, scale):
    """Trace the connected regions of a mask as GeoJSON polygon coordinates (outer ring first, then the holes)."""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    def to_ring(contour):
        # Contour points are pixel centers of the prediction; map them to full resolution pixel coordinates.
        points = (contour[:, 0].astype(float) + 0.5) * scale
        points = np.concatenate([points, points[:1]])
        return [[round(x, 1), round(y, 1)] for x, y in points]

    polygons = []
    for i, (_, _, first_child, parent) in enumerate(hierarchy[0]):
        if parent != -1 or len(contours[i]) < 3:
            continue
        rings = [to_ring(contours[i])]
        child = first_child
        while child != -1:
            if len(contours[child]) >= 3:
                rings.append(to_ring(contours[child]))
            child = hierarchy[0][child][0]
        polygons.append(rings)
    return polygons


def prediction_to_features(prediction, scale):
    features = []
    for class_id, (name, color) in QUPATH_CLASSES.items():
        for rings in mask_to_polygons(prediction == class_id, scale):
            features.append({
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {"type": "Polygon", "coordinates": rings},
                "properties": {"objectType": "annotation", "classification": {"name": name, "color": color}},
            })
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--prediction", required=True, help="Prediction written by predict_tissue_classifier.py.")
    parser.add_argument("-i", "--image", required=True, help="The predicted WSI, to get its full resolution shape.")
    parser.add_argument("-o", "--output", required=True, help="Where to write the annotations (.geojson).")
    args = parser.parse_args()

    with h5py.File(args.prediction, "r") as f:
        prediction = f["prediction"][:]
    full_shape = get_pyramid(args.image)[0][0].shape[:2]
    scale = np.array([full_shape[1] / prediction.shape[1], full_shape[0] / prediction.shape[0]])

    features = prediction_to_features(prediction, scale)
    with open(args.output, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    counts = {name: sum(f["properties"]["classification"]["name"] == name for f in features)
              for name, _ in QUPATH_CLASSES.values()}
    print(f"Saved {len(features)} annotations to {args.output}:", ", ".join(f"{n} {c}" for n, c in counts.items()))


if __name__ == "__main__":
    main()
