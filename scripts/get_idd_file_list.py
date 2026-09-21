import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "redzapdos123/indian-driving-dataset-detections-yolov11"

SOURCE_FILE = "scripts/idd_files.txt"
OUTPUT_FILE = "scripts/idd_required_files.txt"

TRAIN_NEEDED = 300
VAL_NEEDED = 60
TEST_NEEDED = 60


def get_existing_train_test():
    """Take train/test files from the file list we already downloaded."""

    train_files = []
    test_files = []

    with open(SOURCE_FILE, "r") as f:
        for line in f:
            name = line.strip()

            if not name:
                continue

            if name.startswith("IDDDetectionsYOLODataset/train/images/"):
                if len(train_files) < TRAIN_NEEDED:
                    train_files.append(name)

            elif name.startswith("IDDDetectionsYOLODataset/test/images/"):
                if len(test_files) < TEST_NEEDED:
                    test_files.append(name)

            if (
                len(train_files) >= TRAIN_NEEDED
                and len(test_files) >= TEST_NEEDED
            ):
                break

    return train_files, test_files


def get_validation_files():
    """Walk through Kaggle pages until 60 validation images are found."""

    api = KaggleApi()
    api.authenticate()

    val_files = []
    page_token = None
    page_number = 0

    print()
    print("=" * 70)
    print("Searching Kaggle for validation images")
    print("=" * 70)

    while len(val_files) < VAL_NEEDED:

        page_number += 1

        print(
            f"Page {page_number} | "
            f"Validation found: {len(val_files)}/{VAL_NEEDED}"
        )

        try:
            response = api.dataset_list_files(
                DATASET,
                page_size=200,
                page_token=page_token,
            )

        except Exception as e:
            print(f"Kaggle request failed: {e}")
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)
            continue

        for item in response.files:

            name = item.name

            if (
                name.startswith(
                    "IDDDetectionsYOLODataset/val/images/"
                )
                and name.lower().endswith((".jpg", ".jpeg", ".png"))
            ):
                if name not in val_files:
                    val_files.append(name)

                if len(val_files) >= VAL_NEEDED:
                    break

        page_token = response.next_page_token

        if not page_token:
            print("No more pages available.")
            break

        time.sleep(1)

    return val_files


def main():

    print("=" * 70)
    print("NeuroPilots - IDD Required Dataset Files")
    print("=" * 70)

    # ------------------------------------------------------------
    # TRAIN + TEST
    # ------------------------------------------------------------

    print()
    print("Using existing scripts/idd_files.txt for train/test...")

    train_files, test_files = get_existing_train_test()

    print(f"Train selected: {len(train_files)}")
    print(f"Test selected : {len(test_files)}")

    if len(train_files) < TRAIN_NEEDED:
        raise RuntimeError(
            f"Only {len(train_files)} train images available."
        )

    if len(test_files) < TEST_NEEDED:
        raise RuntimeError(
            f"Only {len(test_files)} test images available."
        )

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    val_files = get_validation_files()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"Train : {len(train_files)}")
    print(f"Val   : {len(val_files)}")
    print(f"Test  : {len(test_files)}")

    if len(val_files) < VAL_NEEDED:
        raise RuntimeError(
            f"Only {len(val_files)} validation images found."
        )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    with open(OUTPUT_FILE, "w") as f:

        f.write("# TRAIN IMAGES\n")
        for name in train_files:
            f.write(name + "\n")

        f.write("# VAL IMAGES\n")
        for name in val_files:
            f.write(name + "\n")

        f.write("# TEST IMAGES\n")
        for name in test_files:
            f.write(name + "\n")

    print()
    print(f"Saved required file list to:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()