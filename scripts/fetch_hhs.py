"""Download the Hormone Health Study (OpenNeuro ds005360, CC0) from OpenNeuro's public S3 bucket.

Lists every object under ds005360/, then downloads missing or size-mismatched files with a few
parallel workers. Safe to re-run: completed files are skipped.
"""
import concurrent.futures as cf
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
PREFIX = "ds005360/"
DEST = Path(sys.argv[1] if len(sys.argv) > 1 else "data/hhs")


def list_objects():
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": PREFIX}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        keys = re.findall(r"<Key>(.*?)</Key>", xml)
        sizes = [int(s) for s in re.findall(r"<Size>(\d+)</Size>", xml)]
        out += [(k, s) for k, s in zip(keys, sizes) if not k.startswith(PREFIX + ".datalad")]
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return out
        token = m.group(1)


def fetch(item):
    key, size = item
    path = DEST / key[len(PREFIX):]
    if path.exists() and path.stat().st_size == size:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(key)}", timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size == size:
                tmp.rename(path)
                return size
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"{key}: {err}")


if __name__ == "__main__":
    objs = list_objects()
    total = sum(s for _, s in objs)
    print(f"{len(objs)} files, {total/1e9:.2f} GB", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(6) as ex:
        for n, got in enumerate(ex.map(fetch, objs), 1):
            done += got
            if n % 50 == 0 or n == len(objs):
                print(f"{n}/{len(objs)} files, {done/1e9:.2f} GB downloaded this run", flush=True)
    print("complete", flush=True)
