"""
Load COCO val2017 detection annotations: 5000 images, 80 object categories,
36781 boxes totally, 446 of them iscrowd, 48 images have no boxes.
Downloaded from cocodataset.org on first run (~1GB).

Each box is [x, y, width, height] in pixels, (x, y) = top-left corner.
Category ids go from 1 to 90 with gaps, so id_to_name maps them to names.

Usage:
    python -m src.data
"""

import os
import json
import urllib.request
import zipfile


data_path = os.path.join(os.path.dirname(__file__), "..", "data")
images_url = "http://images.cocodataset.org/zips/val2017.zip"
captions_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

images_zip_path = os.path.join(data_path, "val2017.zip")
captions_zip_path = os.path.join(data_path, "annotations_trainval2017.zip")

images_file = os.path.join(data_path, "val2017")
instances_file = os.path.join(data_path, "annotations", "instances_val2017.json")

def download_and_unzip(url, saved_path, check_path):
    """
    function: download a zip file, unzip it into data/, then delete the zip
    skip if the unzipped file already exists
    """
    if os.path.exists(check_path):
        print(f"Already downloaded: {check_path}")
        return

    os.makedirs(data_path, exist_ok=True)

    urllib.request.urlretrieve(url, saved_path)
    with zipfile.ZipFile(saved_path) as f:
        f.extractall(data_path)
    os.remove(saved_path)



def load_annotations():
    """
    function: read images, boxes and categories from instances_val2017.json

    output:
        images: a list of 5000 dicts, one per image
            [{"id": 397133,
              "path": ".../val2017/000000397133.jpg",
              "boxes": [{"bbox": [217.62, 240.54, 38.99, 57.75],   # [x, y, w, h]
                         "category_id": 44,
                         "category": "bottle",
                         "iscrowd": 0},
                        ...]},
             ...]
        id_to_name: a dict of the 80 categories, {1: "person", 2: "bicycle", ...}
    """
    with open(instances_file, "r") as f:
        coco = json.load(f)     # <class 'dict'>, dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

    id_to_name = {}

    categories = coco["categories"]     # <class 'list'>, dict_keys(['supercategory', 'id', 'name'])
    for category in categories: 
        category_id = category["id"]
        name = category["name"]
        id_to_name[category_id] = name

    images = {}
    for image in coco["images"]:        # <class 'list'>, dict_keys(['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'])
        image_id = image["id"]
        file_name  = image["file_name"]

        path = os.path.join(images_file, file_name)
        images[image_id] = {"id": image_id, "path": path, "boxes": []}


    for annotation in coco["annotations"]:      # <class 'list'> dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        iscrowd = annotation["iscrowd"]

        label = id_to_name[category_id]
        box = {"bbox": bbox, "category_id": category_id, "category": label, "iscrowd": iscrowd}
        images[image_id]["boxes"].append(box)

    images = list(images.values())

    return images, id_to_name

if __name__ == "__main__":
    images, id_to_name = load_annotations()
    print(images[0])
 