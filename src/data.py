"""
Load COCO val2017 detection annotations: images, boxes and the 80 object categories.
Downloaded from cocodataset.org on first run (~1GB).

Each box is [x, y, width, height] in pixels, (x, y) = top-left corner.
Category ids go from 1 to 90 with gaps, so id_to_name maps them to names.

Output:
images: 5000, categories: 80
boxes: 36781 (446 iscrowd)
boxes per image: min 0, max 63; images without boxes: 48
top 5 categories: person 11004, car 1932, chair 1791, book 1161, bottle 1025

Usage (download data and print dataset statistics):
    python -m src.data
"""

import os
import json
import urllib.request
import zipfile
from collections import Counter


data_path = os.path.join(os.path.dirname(__file__), "..", "data")

# images and annotations are two separate zip files
images_url = "http://images.cocodataset.org/zips/val2017.zip"
captions_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# zip files are saved here while downloading
images_zip_path = os.path.join(data_path, "val2017.zip")
captions_zip_path = os.path.join(data_path, "annotations_trainval2017.zip")

# where the files end up after unzipping
images_file = os.path.join(data_path, "val2017")
instances_file = os.path.join(data_path, "annotations", "instances_val2017.json")

def download_and_unzip(url, saved_path, check_path):
    """
    function: download a zip file, unzip it into data/, then delete the zip
    skip if the unzipped file already exists
    """
    # already unzipped before: nothing to do
    if os.path.exists(check_path):
        print(f"Already downloaded: {check_path}")
        return

    # data/ is not in git, so it may not exist after cloning
    os.makedirs(data_path, exist_ok=True)

    urllib.request.urlretrieve(url, saved_path)     # download the zip
    with zipfile.ZipFile(saved_path) as f:
        f.extractall(data_path)                     # unzip into data/
    os.remove(saved_path)                           # the zip is no longer needed



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
        id_to_name[category_id] = name      # {1: "person", 2: "bicycle", ...}

    # use the image id as key
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

        label = id_to_name[category_id]     # 44 -> "bottle"
        box = {"bbox": bbox, "category_id": category_id, "category": label, "iscrowd": iscrowd}
        images[image_id]["boxes"].append(box) 

    # turn the dict into a list
    images = list(images.values())

    return images, id_to_name

def describe(images):
    """
    function: print dataset statistics

    input: images from load_annotations()

    output:
        number of images, boxes (and how many are iscrowd) and categories
        boxes per image: min, max, and how many images have no boxes
        the 5 categories with the most boxes
        the boxes of the first image
    """
    n_boxes = []        # number of boxes of each image
    total_iscrowd = 0   # boxes that cover a group of objects
    n_no_box = 0        # images without any box
    categories = []     # category name of every box

    n_images = len(images)

    # outer loop: images, inner loop: the boxes of one image
    for image in images:
        image_boxes = len(image["boxes"])
        n_boxes.append(image_boxes)

        if image_boxes == 0:
            n_no_box += 1

        for box in image["boxes"]:
            if box["iscrowd"] == 1:
                total_iscrowd += 1
            categories.append(box["category"])

    # count boxes per category, then take the 5 largest: [("person", 11004), ...]
    counts = Counter(categories)
    most_common_categories = counts.most_common(5)

    total_boxes = sum(n_boxes)
    total_categories = len(set(categories))     # set removes duplicates

    print(f"images: {n_images}, categories: {total_categories}")
    print(f"boxes: {total_boxes} ({total_iscrowd} iscrowd)")
    print(f"boxes per image: min {min(n_boxes)}, max {max(n_boxes)}; images without boxes: {n_no_box}")

    print("\ntop 5 categories:")
    for name, count in most_common_categories:
        print(f"  {name}: {count}")

    # show the first image, to check that path and boxes match
    first_image = images[0]
    print(f"\nfirst image: {first_image['path']}")
    for box in first_image["boxes"]:
        print(f"  - {box['category']} {box['bbox']}")


if __name__ == "__main__":
    images, id_to_name = load_annotations()
    describe(images)
 