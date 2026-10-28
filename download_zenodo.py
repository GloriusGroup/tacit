"""Download the TACIT checkpoints and the USPTO-MIT data from Zenodo.

Places the files in the folders the configs expect. Files that are already
present are kept and checked against their Zenodo checksums.
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/23669984/files/{}?download=1"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "uspto_mit"
SPLITS = ("train.jsonl", "val.jsonl", "test.jsonl")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, checksum: str, dest: Path) -> Path:
    path = dest / name
    dest.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {name}")
        part = path.with_name(name + ".part")
        with urllib.request.urlopen(URL.format(name)) as response, open(part, "wb") as f:
            shutil.copyfileobj(response, f)
        part.rename(path)
    if md5(path) != checksum:
        path.unlink()
        sys.exit(f"Checksum mismatch for {name}, run again")
    return path


fetch("tacit.pt", "35ba77776319fcbd3e3d5b46d9cc965b", ROOT / "runs" / "tacit" / "checkpoints")
fetch("tacit_no-stop.pt", "e7a968c6d1f26259080c1ad0bc356b51", ROOT / "runs" / "tacit_no-stop" / "checkpoints")

if not all((DATA / split).exists() for split in SPLITS):
    archive = fetch("uspto_mit_blockspace.zip", "1fec19d0b0f10dec5c54941e055d62ad", DATA)
    print(f"Extracting {archive.name}")
    with zipfile.ZipFile(archive) as z:
        for split in SPLITS:
            with z.open(f"uspto_mit_blockspace/{split}") as src, open(DATA / split, "wb") as dst:
                shutil.copyfileobj(src, dst)
    archive.unlink()

print("Done")
