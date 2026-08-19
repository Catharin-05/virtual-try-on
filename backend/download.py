"""
download_sapiens_weights.py

The sapiens_inference package's built-in downloader (common.py) does a
single raw `requests.get(..., stream=True)` with no retry/resume logic.
For the Sapiens-1B segmentation weights (~4.7GB), any network hiccup kills
the whole download and you have to start from zero -- which is almost
certainly what caused:

    requests.exceptions.ChunkedEncodingError: ('Connection broken:
    IncompleteRead(2149995628 bytes read, 2566318429 more expected)', ...)

This script downloads the same file via `huggingface_hub`, which DOES
support resuming interrupted downloads and retries transient errors, then
places it at the exact path sapiens_inference expects
(./models/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt)
so that the next time you run main.py / segmentation_detector.py, the
library sees the file already there and skips downloading entirely.

Install:
    pip install huggingface_hub

Usage (run from the same directory you'll run main.py from, e.g. backend/):
    python download_sapiens_weights.py
"""

import os
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "facebook/sapiens-seg-1b-torchscript"
FILENAME = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
MODEL_DIR = "models"  # must match sapiens_inference's default model_dir


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_path = os.path.join(MODEL_DIR, FILENAME.replace(".pt2", ".pt"))

    if os.path.exists(final_path):
        print(f"Already present at {final_path} -- nothing to do.")
        return

    print(f"Downloading {FILENAME} from {REPO_ID} "
          f"(resumable -- safe to re-run this script if it drops)...")

    # hf_hub_download resumes partial downloads and retries on its own,
    # unlike the plain requests.get() the library uses internally.
    downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    shutil.copy(downloaded_path, final_path)
    print(f"Done. Placed weights at: {final_path}")
    print("You can now re-run main.py / segmentation_detector.py -- "
          "it will find this file and skip downloading.")


if __name__ == "__main__":
    main()