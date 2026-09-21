import random
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASET = "redzapdos123/indian-driving-dataset-detections-yolov11"

ROOT = Path(__file__).resolve().parent.parent

FILE_LIST = ROOT / "scripts" / "idd_files.txt"

DATA_ROOT = ROOT / "data" / "cv" / "idd"

TRAIN_COUNT = 240
VAL_COUNT = 60
TEST_COUNT = 60

RANDOM_SEED = 42

MAX_RETRIES = 5
RETRY_DELAY = 3


# ============================================================
# Helper functions
# ============================================================

def run_kaggle_download(kaggle_path: str, destination: Path) -> bool:
    """
    Download one file from Kaggle.

    Kaggle creates the downloaded file directly inside destination.
    """

    destination.mkdir(parents=True, exist_ok=True)

    filename = Path(kaggle_path).name
    output_file = destination / filename

    # Do not download again if already present.
    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"      Already exists: {filename}")
        return True

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-f",
        kaggle_path,
        "-p",
        str(destination),
    ]

    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"      Downloading: {filename} "
            f"(attempt {attempt}/{MAX_RETRIES})"
        )

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            print()
            print("ERROR: Kaggle CLI was not found.")
            print("Check with:")
            print()
            print("    kaggle --version")
            print()
            sys.exit(1)

        if result.returncode == 0:
            if output_file.exists() and output_file.stat().st_size > 0:
                return True

        if result.stderr:
            print("      Kaggle error:")
            print(result.stderr.strip())

        if attempt < MAX_RETRIES:
            wait_time = RETRY_DELAY * attempt
            print(f"      Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return False


def image_to_label_path(image_path: str) -> str:
    """
    Convert:

        .../train/images/example.jpg

    into:

        .../train/labels/example.txt
    """

    label_path = image_path.replace(
        "/images/",
        "/labels/",
    )

    label_path = str(Path(label_path).with_suffix(".txt"))

    return label_path


def select_files(files: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Select:

        240 train
         60 validation
         60 test

    Validation is created from the available train images because
    this Kaggle dataset does not expose a val/images directory
    in the file list we collected.
    """

    train_images = [
        f
        for f in files
        if "/train/images/" in f
        and f.lower().endswith(".jpg")
    ]

    test_images = [
        f
        for f in files
        if "/test/images/" in f
        and f.lower().endswith(".jpg")
    ]

    print()
    print("Available files:")
    print(f"    Train images: {len(train_images)}")
    print(f"    Test images : {len(test_images)}")
    print()

    required_train = TRAIN_COUNT + VAL_COUNT

    if len(train_images) < required_train:
        raise RuntimeError(
            f"Need at least {required_train} train images, "
            f"but only {len(train_images)} were found."
        )

    if len(test_images) < TEST_COUNT:
        raise RuntimeError(
            f"Need at least {TEST_COUNT} test images, "
            f"but only {len(test_images)} were found."
        )

    random.seed(RANDOM_SEED)

    random.shuffle(train_images)
    random.shuffle(test_images)

    selected_train = train_images[:TRAIN_COUNT]

    selected_val = train_images[
        TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT
    ]

    selected_test = test_images[:TEST_COUNT]

    return selected_train, selected_val, selected_test


def create_directories() -> None:
    """
    Create the NeuroPilots IDD directory structure.
    """

    for split in ["train", "val", "test"]:

        image_dir = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)


def download_split(split: str, image_files: list[str]) -> tuple[int, int]:
    """
    Download images and corresponding YOLO labels.
    """

    image_destination = DATA_ROOT / "images" / split
    label_destination = DATA_ROOT / "labels" / split

    successful = 0
    failed = 0

    print()
    print("=" * 70)
    print(f"{split.upper()} DATA")
    print("=" * 70)

    for index, image_path in enumerate(image_files, start=1):

        filename = Path(image_path).name

        label_path = image_to_label_path(image_path)

        print()
        print(
            f"[{index}/{len(image_files)}] "
            f"{filename}"
        )

        # ----------------------------------------------------
        # Download image
        # ----------------------------------------------------

        image_ok = run_kaggle_download(
            image_path,
            image_destination,
        )

        # ----------------------------------------------------
        # Download label
        # ----------------------------------------------------

        label_ok = run_kaggle_download(
            label_path,
            label_destination,
        )

        if image_ok and label_ok:
            successful += 1
        else:
            failed += 1
            print(f"      FAILED: {filename}")

    return successful, failed


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("NeuroPilots - IDD Small Dataset Builder")
    print("=" * 70)

    # --------------------------------------------------------
    # Check file list
    # --------------------------------------------------------

    if not FILE_LIST.exists():

        print()
        print("ERROR:")
        print(f"File not found: {FILE_LIST}")
        print()
        print("Expected:")
        print()
        print("    scripts/idd_files.txt")
        print()

        sys.exit(1)

    # --------------------------------------------------------
    # Read Kaggle file list
    # --------------------------------------------------------

    files = [
        line.strip()
        for line in FILE_LIST.read_text().splitlines()
        if line.strip()
    ]

    print()
    print(f"File list loaded: {len(files)} entries")

    # --------------------------------------------------------
    # Select dataset
    # --------------------------------------------------------

    train_files, val_files, test_files = select_files(files)

    print("=" * 70)
    print("SELECTED DATASET")
    print("=" * 70)

    print(f"Train : {len(train_files)} images")
    print(f"Val   : {len(val_files)} images")
    print(f"Test  : {len(test_files)} images")

    print()
    print("Validation split:")
    print(
        "    Validation images are selected from the available "
        "train images."
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Download data
    # --------------------------------------------------------

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    total_success = 0
    total_failed = 0

    for split, image_files in splits.items():

        success, failed = download_split(
            split,
            image_files,
        )

        total_success += success
        total_failed += failed

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IDD SUBSET DOWNLOAD COMPLETE")
    print("=" * 70)

    print(f"Successful image/label pairs : {total_success}")
    print(f"Failed image/label pairs     : {total_failed}")

    print()
    print("Dataset location:")
    print(DATA_ROOT)

    print()
    print("Directory structure:")
    print()
    print("data/cv/idd/")
    print("├── images/")
    print("│   ├── train/")
    print("│   ├── val/")
    print("│   └── test/")
    print("│")
    print("└── labels/")
    print("    ├── train/")
    print("    ├── val/")
    print("    └── test/")
    print()

    if total_failed > 0:
        print("WARNING:")
        print("Some files failed to download.")
        print("You can run this script again; existing files will")
        print("automatically be skipped.")
    else:
        print("All selected files downloaded successfully.")

    print()


if __name__ == "__main__":
    main()