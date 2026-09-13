"""Plain parallel mirror of one pinned HF dataset revision, driven by a size-annotated file list.
Skips files already present with the right size; resumes partial files; verifies size on finish.
    python mirror_fetch.py <tree.txt> <dest_dir> [workers]
"""
import os, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
REV = "07132f15e3c6cc6714ae84835b1896d734c5d54a"
REPO = "JerrrrryL/awm-gsm8k-trajectories"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"
tree, dest, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8
tok = os.environ["MY_HF_TOKEN"]
H = {"Authorization": f"Bearer {tok}"}
files = []
for line in open(tree):
    size, path = line.rstrip("\n").split("\t", 1)
    files.append((int(size), path))
todo = [(s, p) for s, p in files if not (os.path.exists(os.path.join(dest, p)) and os.path.getsize(os.path.join(dest, p)) == s)]
# big files first so the long tail is small files
todo.sort(key=lambda x: -x[0])
total = sum(s for s, _ in todo)
print(f"{len(files)} files in tree; {len(todo)} to fetch, {total/1e9:.1f} GB", flush=True)
lock = threading.Lock(); done_bytes = [0]; done_n = [0]; t0 = time.time()

def fetch(size, path):
    out = os.path.join(dest, path); os.makedirs(os.path.dirname(out), exist_ok=True)
    part = out + ".part"
    for attempt in range(8):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have > size: have = 0; os.remove(part)
            hdr = dict(H); mode = "ab" if have else "wb"
            if have: hdr["Range"] = f"bytes={have}-"
            with requests.get(BASE + path, headers=hdr, stream=True, timeout=(30, 120), allow_redirects=True) as r:
                if r.status_code == 416: pass
                elif r.status_code in (200, 206):
                    if r.status_code == 200 and have: mode = "wb"
                    with open(part, mode) as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                elif r.status_code == 429:
                    time.sleep(15 * (attempt + 1)); continue
                else:
                    raise RuntimeError(f"HTTP {r.status_code}")
            got = os.path.getsize(part)
            if got != size:
                if got > size: os.remove(part)
                raise RuntimeError(f"size {got} != {size}")
            os.replace(part, out)
            with lock:
                done_bytes[0] += size; done_n[0] += 1
                if done_n[0] % 50 == 0 or size > 500_000_000:
                    el = time.time() - t0
                    print(f"{done_n[0]}/{len(todo)} {done_bytes[0]/1e9:.1f}/{total/1e9:.1f} GB {done_bytes[0]/el/1e6:.1f} MB/s  {path}", flush=True)
            return True
        except Exception as e:
            time.sleep(5 * (attempt + 1))
            last = e
    print(f"FAILED {path}: {last}", flush=True); return False

with ThreadPoolExecutor(workers) as ex:
    res = list(ex.map(lambda sp: fetch(*sp), todo))
print(f"DONE ok={sum(res)} failed={len(res)-sum(res)} in {(time.time()-t0)/60:.1f} min", flush=True)
