from pathlib import Path

# Root for this project (edit if you want)
PROJECT_ROOT = Path("/mnt/c/Users/ooluwadare/maskrcnn_ctc").resolve()

# Your CTC dataset (you already confirmed this)
CTC_ROOT = Path("/mnt/c/Users/ooluwadare/stardist-main/data/Fluo-C2DL-MSC").resolve()

# Where your COCO jsons are saved
COCO_DIR = (PROJECT_ROOT / "data" / "coco" / "Fluo-C2DL-MSC").resolve()

# Output directory for detectron2
OUTPUT_DIR = (PROJECT_ROOT / "outputs").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
