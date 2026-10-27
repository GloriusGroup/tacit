#!/usr/bin/env bash
# Downloads the TACIT checkpoints and the USPTO-MIT data from Zenodo into the
# folders the configs expect. Files that are already present are kept.
set -euo pipefail
cd "$(dirname "$0")"

URL=https://zenodo.org/records/23669984/files

fetch() {  # fetch <file> <md5> <destination dir>
  mkdir -p "$3"
  if [ ! -f "$3/$1" ]; then
    echo "Downloading $1"
    curl -#fL --retry 3 -o "$3/$1.part" "$URL/$1?download=1"
    mv "$3/$1.part" "$3/$1"
  fi
  echo "$2  $3/$1" | md5sum -c --quiet || { rm "$3/$1"; echo "Checksum mismatch for $1, run again"; exit 1; }
}

fetch tacit.pt 35ba77776319fcbd3e3d5b46d9cc965b runs/tacit/checkpoints
fetch tacit_no-stop.pt e7a968c6d1f26259080c1ad0bc356b51 runs/tacit_no-stop/checkpoints

if [ ! -f data/uspto_mit/train.jsonl ] || [ ! -f data/uspto_mit/val.jsonl ] || [ ! -f data/uspto_mit/test.jsonl ]; then
  fetch uspto_mit_blockspace.zip 1fec19d0b0f10dec5c54941e055d62ad data/uspto_mit
  unzip -jo data/uspto_mit/uspto_mit_blockspace.zip '*.jsonl' -d data/uspto_mit
  rm data/uspto_mit/uspto_mit_blockspace.zip
fi

echo "Done"
