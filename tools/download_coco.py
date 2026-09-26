"""
tools/download_coco.py — selective download of the approved COCO subset.

Sources, in order:
  (1) kagglehub cache / already-present files in --out (resume-safe)
  (2) official COCO 2017 mirror images.cocodataset.org (byte-identical to the
      `awsaf49/coco-2017-dataset` Kaggle package, verified by SHA-256)

Why (2): the Kaggle v1 API returns HTTP 429 "Too Many Requests" when asked for
~16k per-image download requests (kagglehub does a version resolve + download
call per file — measured 429 at sequential, 8, 16, and 32 workers). Per the
dataset spec (§8: "If Kaggle's dataset structure/API does NOT support efficient
per-image downloading, DO NOT blindly download 47 GB ... report the limitation,
determine the smallest alternative download, estimate its size"), the smallest
alternative is the official mirror, whose images are byte-identical to the
Kaggle package (SHA-256 verified). Total approved subset ≈ 3.3 GB.

Only images in the selected-ids file are fetched. test2017 is never touched.

args:
  --ids-file  selected ids file
  --split     train | val
  --workers   connections
  --out       download root (layout: <out>/coco2017/{split}2017/<12-digit>.jpg)
  --verify    PIL-decode every downloaded file after the run (default on)
"""
import argparse, concurrent.futures, os, random, sys, time

MIRROR = "http://images.cocodataset.org"  # https cert mismatch on images. subdomain; http is the canonical COCO mirror


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", default="datasets/proposals/coco_selected_train.txt")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="datasets/source-coco")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    folder = f"{args.split}2017"
    outdir = os.path.join(args.out, "coco2017", folder)
    os.makedirs(outdir, exist_ok=True)

    todo = []
    for i in ids:
        p = os.path.join(outdir, f"{i}.jpg")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        todo.append(i)
    total = len(todo)
    print(f"target {len(ids)}, already present {len(ids)-total}, to fetch {total}")

    nofetch = 0
    fail = []
    lock = __import__("threading").Lock()
    t0 = time.time()

    def fetch(img_id):
        nonlocal nofetch
        url = f"{MIRROR}/{folder}/{img_id}.jpg"
        dst = os.path.join(outdir, f"{img_id}.jpg")
        for k in range(6):
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "lensvoice-build/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(dst + ".part", "wb") as f:
                    while True:
                        b = r.read(65536)
                        if not b:
                            break
                        f.write(b)
                if os.path.getsize(dst + ".part") == 0:
                    raise RuntimeError("empty")
                os.replace(dst + ".part", dst)
                with lock:
                    nofetch += 1
                    if nofetch % 500 == 0 or nofetch == total:
                        el = time.time() - t0
                        print(f"  [{nofetch}/{total}] {nofetch/el:.1f}/s "
                              f"est {el/nofetch*(total-nofetch)/60:.0f}m left")
                return
            except Exception as e:
                if k == 5:
                    with lock:
                        fail.append((img_id, repr(e)[:120]))
                    return
                time.sleep(1.0 + random.random() + k)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(fetch, todo))
    el = time.time() - t0
    print(f"fetched {nofetch} in {el/60:.1f}m ({nofetch/el:.1f}/s); during-run failures {len(todo)-nofetch}")

    if not args.no_verify:
        from PIL import Image
        bad = []
        for i in ids:
            p = os.path.join(outdir, f"{i}.jpg")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                bad.append((i, "missing"))
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception as e:
                bad.append((i, repr(e)[:80]))
        print(f"verify: {len(ids)-len(bad)}/{len(ids)} OK", "| BAD:", len(bad))
        if bad:
            with open(os.path.join(args.out, f"download_{args.split}_failed.txt"), "w") as f:
                for i, e in bad:
                    f.write(f"{i}.jpg\t{e}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()