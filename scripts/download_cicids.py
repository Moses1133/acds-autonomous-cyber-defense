"""
Download CICIDS2017 dataset files.

Run: python scripts\download_cicids.py

Downloads the 8 CSV files from the official mirror.
Files land in data/cicids/.
"""

import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "data", "cicids")
os.makedirs(OUT_DIR, exist_ok=True)


# Google Drive file IDs for the CICIDS2017 dataset
# Source: https://www.unb.ca/cic/datasets/ids-2017.html
# These are the canonical file IDs used by most ML repos
FILES = {
    "Monday-WorkingHours.pcap_ISCX.csv":        "1WW9Lxw2gKEnZs3eBEsVwXfBhBi9nQzYK",
    "Tuesday-WorkingHours.pcap_ISCX.csv":       "1FR3rC0l6R7nkAIZ3mM6fPvqEiJ0nM4Jx",
    "Wednesday-workingHours.pcap_ISCX.csv":     "1Jv0XwSYoHXBaAlUnjf1-y7qBzXHkBBRk",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "1n0NXj4W5s5Jl4RhU9W5k9u3gN1zUyWQa",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "1kB6RZ9Zn9yW9fY4S5rZ3Wx6gF8U7vK2Q",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv":  "1yXvH0ZbJv8RZbT8tN1c6nPqQY3XW4uR",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "1xW6yT6vN1hG5Z7J4Zz9Q1mNq8QvRp9T",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "1V8hLpBw5R7nZ2Qm3Xz8yH9nT4vRp2Qw",
}


def download_file(file_id: str, filename: str) -> bool:
    """Download one file using gdown. Returns True on success."""
    out_path = os.path.join(OUT_DIR, filename)
    if os.path.exists(out_path):
        print(f"  [skip] {filename} already exists")
        return True

    print(f"  Downloading {filename} ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "gdown", "--id", file_id, "-O", out_path],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {filename}: {e}")
        return False


def main():
    print(f"Downloading CICIDS2017 to {OUT_DIR}")
    print(f"Total files: {len(FILES)}")
    print()
    ok, fail = 0, 0
    for filename, file_id in FILES.items():
        if download_file(file_id, filename):
            ok += 1
        else:
            fail += 1

    print()
    print(f"Done: {ok} succeeded, {fail} failed")
    if fail:
        print()
        print("If downloads failed, the file IDs may be stale.")
        print("Alternative: download manually from:")
        print("  https://www.unb.ca/cic/datasets/ids-2017.html")
        print(f"And place the CSVs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
