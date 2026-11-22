"""Fetch the exact external assets recorded for the Panda experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT/"assets/robotics")
    a = p.parse_args()
    manifest = json.loads((ROOT/"configs/robotics_assets.json").read_text())
    downloads = a.root/"downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for item in manifest["assets"]:
        path = downloads/item["filename"]
        if not path.exists():
            print("Downloading", item["name"], flush=True)
            temporary = path.with_suffix(path.suffix+".partial")
            urllib.request.urlretrieve(item["url"], temporary)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}.")
        if path.name.endswith(".tar.gz"):
            dest = a.root/"vendor"/item["name"]
            if dest.exists():
                continue
            staging = downloads/(item["name"]+"_extracted")
            staging.mkdir(exist_ok=True)
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    target = (staging/member.name).resolve()
                    if not target.is_relative_to(staging.resolve()) or member.issym() or member.islnk():
                        raise ValueError("Unexpected archive member: "+member.name)
                archive.extractall(staging)
            folders = list(staging.iterdir())
            if len(folders) != 1 or not folders[0].is_dir():
                raise ValueError("Unexpected source archive structure")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(folders[0]), dest)
            staging.rmdir()
        print("Verified", item["name"], flush=True)

if __name__ == "__main__":
    main()
