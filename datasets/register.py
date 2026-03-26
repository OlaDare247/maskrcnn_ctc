from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from .paths import COCO_DIR

def register_msc():
    """
    Registers Fluo-C2DL-MSC datasets using the COCO jsons you created.
    """
    name_train = "msc01_train_ST"
    name_val   = "msc01_val_GT"

    train_json = str(COCO_DIR / "msc01_train_ST.json")
    val_json   = str(COCO_DIR / "msc01_val_GT.json")

    # image_root is inside the json already (file_name is absolute or relative).
    # If your json uses absolute file paths: image_root can be "".
    # If your json uses relative paths: set image_root to the dataset root.
    image_root = ""  # safest if json has absolute file paths

    register_coco_instances(name_train, {}, train_json, image_root)
    register_coco_instances(name_val,   {}, val_json,   image_root)

    # Optional: set class names for visualization
    MetadataCatalog.get(name_train).thing_classes = ["cell"]
    MetadataCatalog.get(name_val).thing_classes = ["cell"]

    return name_train, name_val
