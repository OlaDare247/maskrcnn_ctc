import copy
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


def _ensure_hwc3(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is HxWx3 uint8.
    Handles weird shapes from TIFF (e.g., (1,H,W,1), (H,W,1,1), (1,H,W), etc.)
    """
    image = np.asarray(image)
    image = np.squeeze(image)  # remove singleton dims

    # Now allow only 2D or 3D
    if image.ndim == 2:
        image = image[:, :, None]  # HxWx1

    elif image.ndim == 3:
        # If CHW, convert to HWC (rare but possible)
        if image.shape[0] in (1, 3) and image.shape[2] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        # If still not HWC-like, let it fall through and error
    else:
        raise ValueError(f"Unsupported image ndim after squeeze: {image.ndim}, shape={image.shape}")

    # At this point image is HxWxC
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] >= 3:
        image = image[:, :, :3]
    else:
        raise ValueError(f"Bad channel dimension after processing: shape={image.shape}")

    # Convert to uint8 for Detectron2 augmentations
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def build_mapper(cfg, is_train=True):
    """
    Returns a mapper fn used by dataloader.
    Converts grayscale/TIFF weird shapes -> HxWx3 + standard augmentations.
    """
    augmentations = []
    if is_train:
        augmentations.extend([
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomFlip(horizontal=False, vertical=True),
        ])
    return lambda d: _mapper(d, augmentations, is_train=is_train)


def _mapper(dataset_dict, augmentations, is_train=True):
    dataset_dict = copy.deepcopy(dataset_dict)

    # Read grayscale (may come back with odd dims for TIFF)
    image = utils.read_image(dataset_dict["file_name"], format="L")

    # Force HxWx3 uint8
    image = _ensure_hwc3(image)

    # Augment
    aug_input = T.AugInput(image)
    transforms = T.AugmentationList(augmentations)(aug_input)
    image = aug_input.image

    # To tensor CHW float32
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    if not is_train:
        return dataset_dict

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.get("annotations", [])
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2], mask_format='bitmask')
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict




# import copy
# import numpy as np
# import torch

# from detectron2.data import detection_utils as utils
# from detectron2.data import transforms as T

# def build_mapper(cfg, is_train=True):
#     """
#     Returns a mapper fn used by dataloader.
#     Converts grayscale -> 3-channel + standard augmentations.
#     """
#     augmentations = []
#     if is_train:
#         augmentations.extend([
#             T.RandomFlip(horizontal=True, vertical=False),
#             T.RandomFlip(horizontal=False, vertical=True),
#         ])
#     # Resize policy: keep Detectron2 defaults from cfg
#     return lambda d: _mapper(d, augmentations, is_train=is_train)

# def _mapper(dataset_dict, augmentations, is_train=True):
#     dataset_dict = copy.deepcopy(dataset_dict)

#     image = utils.read_image(dataset_dict["file_name"], format="L")  # grayscale
#     # Convert to 3-channel
#     image = np.repeat(image[:, :, None], 3, axis=2)

#     aug_input = T.AugInput(image)
#     transforms = T.AugmentationList(augmentations)(aug_input)
#     image = aug_input.image

#     image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
#     dataset_dict["image"] = image

#     if not is_train:
#         return dataset_dict

#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[1:])
#         for obj in dataset_dict.get("annotations", [])
#         if obj.get("iscrowd", 0) == 0
#     ]
#     instances = utils.annotations_to_instances(annos, image.shape[1:])
#     dataset_dict["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dict
    
#     image = _ensure_hwc3(image)
#     aug_input = T.AugInput(image)

#     aug_input = T.AugInput(image)
#     transforms = T.AugmentationList(augmentations)(aug_input)
#     image = aug_input.image

#     image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
#     dataset_dict["image"] = image

#     if not is_train:
#         return dataset_dict

#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[1:])
#         for obj in dataset_dict.get("annotations", [])
#         if obj.get("iscrowd", 0) == 0
#     ]
#     instances = utils.annotations_to_instances(annos, image.shape[1:])
#     dataset_dict["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dict
