"""exl3_experts_pack.py: the per-rank, experts-only EXL3 folder that TF_FLASHNEXT_EXPERTS_EXL3=DIR opens, cut from an EXL3
Flash Next checkpoint (turboderp/Qwen3.8-Flash-Next-exl3) for two ranks.

  fetch    --repo R --rev SHA --shards N --out DIR   take only the routed-expert tensors (and the MTP head's) from the
                                                     pinned shards by HTTP range (curl -r): ~37.5 GiB of the 2.68 bpw
                                                     branch instead of its whole shards
  extract  --src DIR --out DIR                       the same files cut from whole shards already on disk (model-*.safetensors)
  verify   --out DIR --pins FILE [--ranks 0]         every file of those ranks, and the index, against the pinned sha256
  plan     (--src DIR | --repo R --rev SHA --shards N)   per rank: tensors, bytes, files; fetches headers only

Common options: --ranks 0,1 (the ranks whose files to write; the index always names both), --pins FILE (pinned sha256s:
each source shard's header and, with extract --check-shards, the whole shard), --shard-gib 4, --chunk-mib 256,
--limit-rate 0 (curl --limit-rate; 0: none).

The layout (identical in fetch and extract): rank 0 holds routed experts 0-255 of every decoder layer plus every MTP-head
expert, rank 1 experts 256-511; a tensor is model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.{trellis,suh,
svh,mul1} or mtp.layers.0.mlp.experts.E.{...}, under the checkpoint's own name and dtype. Each rank's tensors, sorted by
source shard and offset, go into experts-r<R>-0000i-of-0000n.safetensors files of at most --shard-gib, beside one
model.safetensors.index.json for both ranks, manifest.json (per file: sha256, tensors, bytes) and SHA256SUMS. The engine
reads the index, skips files that are not there, and loads the experts its rank owns, so each Spark needs only its own
rank's files.

The checkpoint is untrusted data: only safetensors JSON headers are parsed and byte ranges copied; nothing from it is
imported or run. Every step resumes: rerun the same command after an interruption.
"""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time

PAT = re.compile(r"^(model\.language_model\.|mtp\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mul1)$")
DTYPES = {"F16": 2, "I16": 2, "I32": 4, "BF16": 2, "F32": 4}
MAX_HEADER = 256 << 20
WORLD, EXPERTS, MTP_RANK = 2, 512, 0


def log(out, msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg
    print(line, flush=True)
    if out:
        with open(os.path.join(out, "fetch.log"), "a") as f:
            f.write(line + "\n")


def read_pins(path):
    pins = {"header": {}, "shard": {}, "file": {}, "index": None, "manifest": None, "sums": None}
    if not path:
        return pins
    for raw in open(path):
        w = raw.split()
        if not w or w[0].startswith("#"):
            continue
        kind, sha, name = w[0], w[1], w[2]
        if kind in ("header", "shard", "file"):
            pins[kind][name] = (sha, int(w[3]) if len(w) > 3 else None)
        elif kind in ("index", "manifest", "sums"):
            pins[kind] = sha
        else:
            sys.exit(f"{path}: unknown pin kind {kind}")
    return pins


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(64 << 20)
            if not c:
                return h.hexdigest()
            h.update(c)


def curl_range(url, a, b, dest, rate):
    want = b - a
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have > want:
        os.remove(dest)
        have = 0
    for attempt in range(8):
        if have == want:
            return
        cmd = ["curl", "-sS", "-L", "--fail", "--proto", "=https", "--proto-redir", "=https",
               "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
               "--max-filesize", str(max(want - have, 0) + 65536), "-r", f"{a + have}-{b - 1}", url]
        if rate and rate != "0":
            cmd[1:1] = ["--limit-rate", rate]
        with open(dest, "ab") as f:
            subprocess.run(cmd, stdout=f, check=False)
        have = os.path.getsize(dest)
        if have > want:
            os.remove(dest)
            sys.exit(f"range {a}-{b} of {url}: got {have} bytes, more than {want}: the server ignored the range")
        if have == want:
            return
        time.sleep(min(300, 15 * 2 ** attempt))
    sys.exit(f"range {a}-{b} of {url}: {have}/{want} bytes after retries")


def shard_names(n):
    return [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]


def check_header(name, raw, pins):
    pin = pins["header"].get(name)
    if pin and hashlib.sha256(raw).hexdigest() != pin[0]:
        sys.exit(f"{name}: its header differs from the pinned one (scripts/exl3-pack.sha256): another revision?")


def remote_headers(args, pins, work):
    hd = os.path.join(work, "hdr")
    os.makedirs(hd, exist_ok=True)
    out, urls = {}, {}
    for name in shard_names(args.shards):
        url = f"https://huggingface.co/{args.repo}/resolve/{args.rev}/{name}"
        urls[name] = url
        lp = os.path.join(hd, name + ".len")
        curl_range(url, 0, 8, lp, args.limit_rate)
        n = struct.unpack("<Q", open(lp, "rb").read())[0]
        if not 0 < n < MAX_HEADER:
            sys.exit(f"{name}: header length {n} out of bounds")
        hp = os.path.join(hd, name + ".json")
        curl_range(url, 8, 8 + n, hp, args.limit_rate)
        raw = open(hp, "rb").read()
        check_header(name, raw, pins)
        out[name] = (json.loads(raw), None, 8 + n)
    return out, urls


def local_headers(src, n, pins):
    out = {}
    for name in shard_names(n):
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            sys.exit(f"{path}: missing")
        with open(path, "rb") as f:
            k = struct.unpack("<Q", f.read(8))[0]
            if not 0 < k < MAX_HEADER:
                sys.exit(f"{name}: header length {k} out of bounds")
            raw = f.read(k)
        check_header(name, raw, pins)
        out[name] = (json.loads(raw), path, 8 + k)
    return out


def select(headers, rank):
    per = EXPERTS // WORLD
    lo, hi = rank * per, (rank + 1) * per
    out = []
    for src, (hdr, path, base) in headers.items():
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            m = PAT.match(k)
            if not m:
                continue
            stack, _, e, _, _ = m.groups()
            e = int(e)
            if stack == "mtp.":
                if rank != MTP_RANK:
                    continue
            elif not lo <= e < hi:
                continue
            a, b = v["data_offsets"]
            if v["dtype"] not in DTYPES or a < 0 or b < a or math.prod(v["shape"]) * DTYPES[v["dtype"]] != b - a:
                sys.exit(f"{k}: dtype, shape and size do not agree")
            out.append((src, base, a, b, k, v))
    return out, (lo, hi)


def widths(sel):
    w = {}
    for *_, k, v in sel:
        if k.endswith(".trellis"):
            key = "mtp" if k.startswith("mtp.") else "main"
            w.setdefault(key, {}).setdefault(str(v["shape"][-1] / 16), 0)
            w[key][str(v["shape"][-1] / 16)] += 1
    return w


def layout(headers, shard_gib):
    cap = int(shard_gib * 2**30)
    ranks = {}
    for rank in range(WORLD):
        sel, (lo, hi) = select(headers, rank)
        sel.sort(key=lambda t: (t[0], t[2]))
        groups, cur, size = [], [], 0
        for t in sel:
            if cur and size + (t[3] - t[2]) > cap:
                groups.append(cur)
                cur, size = [], 0
            cur.append(t)
            size += t[3] - t[2]
        if cur:
            groups.append(cur)
        files = []
        for i, group in enumerate(groups):
            hdr, off, places = {}, 0, []
            for src, base, a, b, k, v in group:
                hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + b - a]}
                places.append((src, base, a, b, off))
                off += b - a
            raw = json.dumps(hdr, separators=(",", ":")).encode()
            raw += b" " * (-len(raw) % 8)
            name = f"experts-r{rank}-{i + 1:05d}-of-{len(groups):05d}.safetensors"
            files.append({"file": name, "raw": raw, "data": off, "places": places, "tensors": len(group)})
        ranks[rank] = {"first": lo, "count": hi - lo, "mtp": rank == MTP_RANK, "widths": widths(sel), "files": files}
    return ranks


def write_index(out, ranks, source):
    wm = {}
    for r in ranks.values():
        for f in r["files"]:
            for k in json.loads(f["raw"]):
                wm[k] = f["file"]
    tmp = os.path.join(out, "model.safetensors.index.json.tmp")
    with open(tmp, "w") as o:
        json.dump({"metadata": {"source": source, "experts_only": True}, "weight_map": wm}, o)
    os.replace(tmp, os.path.join(out, "model.safetensors.index.json"))


def prepare_outputs(out, ranks, want):
    for rank in want:
        for f in ranks[rank]["files"]:
            p = os.path.join(out, f["file"])
            total = 8 + len(f["raw"]) + f["data"]
            head = struct.pack("<Q", len(f["raw"])) + f["raw"]
            if os.path.exists(p) and os.path.getsize(p) == total:
                with open(p, "rb") as h:
                    if h.read(len(head)) == head:
                        continue
                sys.exit(f"{p} exists with a different header; refusing to overwrite it (move it away)")
            with open(p, "wb") as h:
                h.write(head)
                h.truncate(total)


def write_manifest(out, ranks, repo, rev):
    m = {"repo": repo, "revision": rev, "world": WORLD, "experts": EXPERTS, "mtp_rank": MTP_RANK, "ranks": {}}
    for rank, r in ranks.items():
        files = []
        for f in r["files"]:
            p = os.path.join(out, f["file"])
            files.append({"file": f["file"], "sha256": sha256_file(p) if os.path.exists(p) else None,
                          "tensors": f["tensors"], "bytes": 8 + len(f["raw"]) + f["data"]})
        m["ranks"][str(rank)] = {"first": r["first"], "count": r["count"], "mtp": r["mtp"], "widths": r["widths"],
                                 "files": files}
    with open(os.path.join(out, "manifest.json.tmp"), "w") as o:
        json.dump(m, o, indent=1)
    os.replace(os.path.join(out, "manifest.json.tmp"), os.path.join(out, "manifest.json"))
    with open(os.path.join(out, "SHA256SUMS"), "w") as o:
        for r in m["ranks"].values():
            for f in r["files"]:
                if f["sha256"]:
                    o.write(f"{f['sha256']}  {f['file']}\n")


def pinned_sums(pins, ranks):
    if not all(pins["file"].get(f["file"]) for r in ranks.values() for f in r["files"]):
        return None
    m = {}
    for rank, r in ranks.items():
        m[rank] = {f["file"]: pins["file"][f["file"]][0] for f in r["files"]}
    return m


def finish(args, ranks, want, pins):
    have = {rank: all(os.path.exists(os.path.join(args.out, f["file"])) for f in ranks[rank]["files"]) for rank in ranks}
    if all(have.values()):
        write_manifest(args.out, ranks, args.repo, args.rev)
        log(args.out, "manifest.json and SHA256SUMS written")
    else:
        sums = pinned_sums(pins, ranks)
        if sums:
            with open(os.path.join(args.out, "SHA256SUMS"), "w") as o:
                for rank in ranks:
                    for name, sha in sums[rank].items():
                        o.write(f"{sha}  {name}\n")
            log(args.out, f"SHA256SUMS written from the pins (only ranks {want} are here)")
    if pins["file"]:
        verify_files(args.out, ranks, want, pins)


def verify_files(out, ranks, want, pins):
    bad = []
    for rank in want:
        for f in ranks[rank]["files"]:
            p = os.path.join(out, f["file"])
            pin = pins["file"].get(f["file"])
            if not pin:
                bad.append(f"{f['file']}: no pin")
                continue
            if not os.path.exists(p) or (pin[1] is not None and os.path.getsize(p) != pin[1]):
                bad.append(f"{f['file']}: missing or of another size")
                continue
            got = sha256_file(p)
            print(f"{f['file']}: {'OK' if got == pin[0] else 'MISMATCH'}", flush=True)
            if got != pin[0]:
                bad.append(f"{f['file']}: sha256 {got}, pinned {pin[0]}")
    ip = os.path.join(out, "model.safetensors.index.json")
    if pins["index"]:
        got = sha256_file(ip) if os.path.exists(ip) else "missing"
        print(f"model.safetensors.index.json: {'OK' if got == pins['index'] else 'MISMATCH'}", flush=True)
        if got != pins["index"]:
            bad.append(f"model.safetensors.index.json: sha256 {got}, pinned {pins['index']}")
    if bad:
        sys.exit("verify failed:\n  " + "\n  ".join(bad))


def fetch(args, pins):
    os.makedirs(os.path.join(args.out, "state"), exist_ok=True)
    headers, urls = remote_headers(args, pins, args.out)
    ranks = layout(headers, args.shard_gib)
    write_index(args.out, ranks, f"{args.repo}@{args.rev}")
    by = {}
    for rank in args.ranks:
        for f in ranks[rank]["files"]:
            if args.only and f["file"] != args.only:
                continue
            dbase = 8 + len(f["raw"])
            for src, _, a, b, off in f["places"]:
                by.setdefault(src, []).append((a, b, f["file"], dbase + off))
    step = args.chunk_mib << 20
    plan = []
    for src in sorted(by):
        runs = []
        for t in sorted(by[src]):
            if runs and t[0] == runs[-1][1] and runs[-1][1] - runs[-1][0] + (t[1] - t[0]) <= step:
                runs[-1][1] = t[1]
                runs[-1][2].append(t)
            else:
                runs.append([t[0], t[1], [t]])
        plan += [(src, a, b, ts) for a, b, ts in runs]
    total = sum(b - a for _, a, b, _ in plan)
    log(args.out, f"plan: ranks {args.ranks}, {len(plan)} ranges, {total / 2**30:.2f} GiB")
    if args.index_only:
        return
    want = [r for r in args.ranks]
    if args.only:
        for rank in want:
            ranks[rank]["files"] = [f for f in ranks[rank]["files"] if f["file"] == args.only] or ranks[rank]["files"]
    prepare_outputs(args.out, ranks, want)
    done_path = os.path.join(args.out, "state", "done.txt")
    done = set(open(done_path).read().split("\n")) if os.path.exists(done_path) else set()
    part, keyf = os.path.join(args.out, "part.bin"), os.path.join(args.out, "state", "part.key")
    got, t0 = sum(b - a for s, a, b, _ in plan if f"{s} {a} {b}" in done), time.time()
    got0 = got
    for i, (src, a, b, ts) in enumerate(plan):
        key = f"{src} {a} {b}"
        if key in done:
            continue
        if not (os.path.exists(keyf) and open(keyf).read() == key) and os.path.exists(part):
            os.remove(part)
        with open(keyf, "w") as k:
            k.write(key)
        base = headers[src][2]
        curl_range(urls[src], base + a, base + b, part, args.limit_rate)
        if os.path.getsize(part) != b - a:
            sys.exit(f"{key}: {os.path.getsize(part)} bytes, not {b - a}")
        handles = {}
        with open(part, "rb") as pf:
            for ta, tb, fname, dst in ts:
                h = handles.get(fname) or handles.setdefault(fname, open(os.path.join(args.out, fname), "r+b"))
                pf.seek(ta - a)
                data = pf.read(tb - ta)
                if len(data) != tb - ta:
                    sys.exit(f"short read of the fetched range at {ta}")
                h.seek(dst)
                h.write(data)
        for h in handles.values():
            h.flush()
            os.fsync(h.fileno())
            h.close()
        with open(done_path, "a") as d:
            d.write(key + "\n")
            d.flush()
            os.fsync(d.fileno())
        os.remove(part)
        got += b - a
        dt = time.time() - t0
        rate = (got - got0) / dt if dt > 0 else 0
        eta = (total - got) / rate if rate > 0 else 0
        log(args.out, f"range {i + 1}/{len(plan)} {src} [{a},{b}) ok; {got / 2**30:.2f}/{total / 2**30:.2f} GiB,"
            f" {rate / 1e6:.1f} MB/s, eta {eta / 60:.0f} min")
    if args.only:
        if pins["file"]:
            verify_files(args.out, ranks, want, pins)
        return
    finish(args, ranks, want, pins)


def extract(args, pins):
    os.makedirs(args.out, exist_ok=True)
    if args.check_shards:
        for name in shard_names(args.shards):
            pin = pins["shard"].get(name)
            if not pin:
                sys.exit(f"{name}: no shard pin in {args.pins}")
            got = sha256_file(os.path.join(args.src, name))
            print(f"{name}: {'OK' if got == pin[0] else 'MISMATCH'}", flush=True)
            if got != pin[0]:
                sys.exit(f"{name}: sha256 {got}, pinned {pin[0]} (re-download it)")
    headers = local_headers(args.src, args.shards, pins)
    ranks = layout(headers, args.shard_gib)
    write_index(args.out, ranks, f"{args.repo}@{args.rev}")
    prepare_outputs(args.out, ranks, args.ranks)
    for rank in args.ranks:
        for f in ranks[rank]["files"]:
            p = os.path.join(args.out, f["file"])
            dbase = 8 + len(f["raw"])
            with open(p, "r+b") as o:
                handles = {}
                for src, base, a, b, off in f["places"]:
                    s = handles.get(src) or handles.setdefault(src, open(headers[src][1], "rb"))
                    s.seek(base + a)
                    o.seek(dbase + off)
                    left = b - a
                    while left:
                        chunk = s.read(min(left, 64 << 20))
                        if not chunk:
                            sys.exit(f"short read in {src}")
                        o.write(chunk)
                        left -= len(chunk)
                for s in handles.values():
                    s.close()
            log(args.out, f"rank {rank}: wrote {f['file']} ({f['tensors']} tensors)")
    finish(args, ranks, args.ranks, pins)


def plan(args, pins):
    if args.src:
        headers = local_headers(args.src, args.shards, pins)
    else:
        work = args.out or os.path.join(os.getcwd(), "exl3-plan")
        os.makedirs(work, exist_ok=True)
        headers, _ = remote_headers(args, pins, work)
    ranks = layout(headers, args.shard_gib)
    for rank, r in ranks.items():
        size = sum(f["data"] for f in r["files"])
        print(f"rank {rank}: experts {r['first']}-{r['first'] + r['count'] - 1}{' + MTP' if r['mtp'] else ''},"
              f" {sum(f['tensors'] for f in r['files'])} tensors, {size / 2**30:.2f} GiB in {len(r['files'])} files;"
              f" widths {r['widths']}")
        for f in r["files"]:
            print(f"  {f['file']} {8 + len(f['raw']) + f['data']} bytes, {f['tensors']} tensors")
    if args.out:
        write_index(args.out, ranks, f"{args.repo}@{args.rev}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("fetch", "extract", "verify", "plan"))
    p.add_argument("--repo", default="turboderp/Qwen3.8-Flash-Next-exl3")
    p.add_argument("--rev")
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--src")
    p.add_argument("--out")
    p.add_argument("--pins")
    p.add_argument("--ranks", default="0,1")
    p.add_argument("--shard-gib", type=float, default=4.0)
    p.add_argument("--chunk-mib", type=int, default=256)
    p.add_argument("--limit-rate", default="0")
    p.add_argument("--only", help="fetch: only this output file (a check of one file against its pin)")
    p.add_argument("--index-only", action="store_true", help="fetch: headers and model.safetensors.index.json only")
    p.add_argument("--check-shards", action="store_true", help="extract: check each whole shard against its pin first")
    args = p.parse_args()
    args.ranks = sorted({int(r) for r in args.ranks.split(",")})
    if not set(args.ranks) <= set(range(WORLD)):
        sys.exit(f"--ranks: 0 and/or 1, not {args.ranks}")
    pins = read_pins(args.pins)
    if args.mode in ("fetch", "extract", "plan") and (not args.rev or not re.fullmatch(r"[0-9a-f]{40}", args.rev)):
        sys.exit("--rev must be a full commit sha (a branch can move)")
    if args.mode == "fetch":
        if not args.out:
            sys.exit("fetch needs --out")
        args.out = os.path.abspath(os.path.expanduser(args.out))
        fetch(args, pins)
    elif args.mode == "extract":
        if not (args.src and args.out):
            sys.exit("extract needs --src and --out")
        extract(args, pins)
    elif args.mode == "verify":
        if not (args.out and args.pins):
            sys.exit("verify needs --out and --pins")
        idx = os.path.join(args.out, "model.safetensors.index.json")
        if not os.path.exists(idx):
            sys.exit(f"{idx}: missing")
        files = sorted(set(json.load(open(idx))["weight_map"].values()))
        ranks = {r: {"files": [{"file": f} for f in files if f.startswith(f"experts-r{r}-")]} for r in range(WORLD)}
        verify_files(args.out, ranks, args.ranks, pins)
    else:
        plan(args, pins)


if __name__ == "__main__":
    main()
