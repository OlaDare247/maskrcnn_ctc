import numpy as np
import tifffile as tiff
from pathlib import Path

def save_label_tiff(out_path: Path, label_img: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # CTC label masks are typically uint16
    tiff.imwrite(str(out_path), label_img.astype(np.uint16))
