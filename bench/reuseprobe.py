#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.request

USAGE = """reuseprobe.py: prompt-reuse probes against an OpenAI-compatible endpoint, one request at a time, thinking off,
temperature 0, text from LCQ v2's haystacks (lcqgen.py build, manifest.json), so every prompt is the same bytes on
every server and every run.

  reuseprobe.py warm|fork|ret OUT.jsonl --url URL --model MODEL [--label LABEL]
  reuseprobe.py report OUT.jsonl [OUT.jsonl ...]     one line a file (the kind from its name: .warm/.lcqw, .fork, .ret/.pcret)

warm  same document, new question (prompt-cache reuse): LCQ v2 at depths LCQ_DEPTHS (131072 262144), one haystack a
      depth (LCQ_HAYS 1): look0 cold, then LCQ_WARM_TASKS (4) other questions on the same haystack, no sampled passes.
      Measured: cached tokens and first-token time of each warm request; one resumed when cached >= 90% of its prompt.
fork  shared opening, parted early (shared-prefix keep): RP_FORK_SETS (2) sets; a set is one opening (the first
      RP_FORK_HEAD (0.70) of haystack h32768-<set>, 22,825 tokens) and RP_FORK_WORKERS (6) workers, each the
      opening, a worker line and its own body (a third of h8192-0 or h8192-1, about 2,680 tokens, so the worker parts
      from the others well before a 512-token tail), then 'Reply OK', max 8 tokens. Measured: cached tokens of each
      worker; worker 1 reads cold, worker 2 parts from worker 1's kept state, workers 3.. resume the opening only when
      the server kept a state where worker 2 parted (one resumed when cached >= 90% of the opening).
ret   prompt-cache retention (cache size): pass 1 reads RP_RET_ITEMS (h400000-0..3 h330000-0..3 h262144-0: about
      3.18M tokens) with one lookup each; pass 2 asks the first RP_RET_BACK (4) of them, the oldest, a second lookup.
      Measured: pass 2's cached tokens (a document came back warm when cached >= 90% of its tokens) and first-token
      times, and the numeric cache fields of URL/health before and after pass 1 when the server has that route.
Environment: RP_TIMEOUT (1800 s a request); LCQ_DEPTHS, LCQ_HAYS, LCQ_WARM_TASKS for warm; LCQ_MANIFEST.
Exit 0 when every request answered, 1 when one failed, 2 on bad arguments."""

HERE = os.path.dirname(os.path.realpath(__file__))
sys.dont_write_bytecode = True
os.environ["LCQ_TIMEOUT"] = os.environ.get("RP_TIMEOUT") or "1800"
spec = importlib.util.spec_from_file_location("lcq", os.path.join(HERE, "lcq.py"))
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
G = L.G
LABEL = ""
fails = 0


def health():
    try:
        with urllib.request.urlopen(L.URL + "/health", timeout=20) as r:
            d = json.loads(r.read().decode())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    flat = {}

    def walk(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{p}.{k}" if p else k)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            flat[p] = x
    walk(d, "")
    return {k: v for k, v in flat.items() if any(s in k.lower() for s in ("cache", "kept", "state", "evict", "prompt_tokens", "cached"))}


def send(f, rec, content, max_tokens):
    global fails
    rec.update(label=LABEL, at=round(time.time(), 3))
    try:
        o = L.ask(content, max_tokens, False)
        u = o["usage"] or {}
        rec.update(prompt_tokens=u.get("prompt_tokens"), cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   completion_tokens=u.get("completion_tokens"), ttft_s=o["ttft_s"], total_s=o["total_s"], prefill_s=o["prefill_s"],
                   finish=o["finish"], text=o["text"][:120])
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        fails += 1
    f.write(json.dumps(rec) + "\n")
    f.flush()
    print(f"{rec['probe']} {rec.get('set', rec.get('pass', ''))} {rec.get('worker', rec.get('item', ''))} tokens {rec.get('prompt_tokens')} "
          f"cached {rec.get('cached_tokens')} ttft {rec.get('ttft_s')}{' ' + rec['error'] if rec.get('error') else ''}", file=sys.stderr, flush=True)
    return rec


def fork(out, by):
    sets = int(os.environ.get("RP_FORK_SETS") or "2")
    workers = int(os.environ.get("RP_FORK_WORKERS") or "6")
    share = float(os.environ.get("RP_FORK_HEAD") or "0.70")
    bodies = []
    for bid in ("h8192-0", "h8192-1"):
        pre, _ = G.build(by[bid])
        n = len(pre) // 3
        bodies += [pre[i * n:(i + 1) * n] for i in range(3)]
    salt = hashlib.sha256(f"{LABEL} {time.time()}".encode()).hexdigest()[:12]
    with open(out, "a") as f:
        for s in range(sets):
            it = by[f"h32768-{s}"]
            pre, _ = G.build(it)
            head = pre[: int(len(pre) * share)]
            head_tokens = int(it["tokens"] * share)
            for w in range(workers):
                body = bodies[w % len(bodies)]
                content = head + f"\n\nworker {w + 1} of set {s} (run {salt})\n" + body + "\n\n" + G.question({})
                send(f, {"probe": "fork", "set": s, "worker": w + 1, "head_tokens_est": head_tokens}, content, 8)
    return 1 if fails else 0


def ret(out, by):
    items = (os.environ.get("RP_RET_ITEMS") or "h400000-0 h400000-1 h400000-2 h400000-3 h330000-0 h330000-1 h330000-2 h330000-3 h262144-0").split()
    back = int(os.environ.get("RP_RET_BACK") or "4")
    built = {}
    with open(out, "a") as f:
        f.write(json.dumps({"probe": "ret", "health": "before", "values": health(), "at": round(time.time(), 3)}) + "\n")
        for i in items:
            pre, tasks = G.build(by[i])
            built[i] = (pre, tasks)
            t = tasks[0]
            send(f, {"probe": "ret", "pass": 1, "item": i, "doc_tokens": by[i]["tokens"]}, pre + G.question(t), t.get("max_tokens", 48))
        f.write(json.dumps({"probe": "ret", "health": "after1", "values": health(), "at": round(time.time(), 3)}) + "\n")
        for i in items[:back]:
            pre, tasks = built[i]
            t = tasks[1]
            send(f, {"probe": "ret", "pass": 2, "item": i, "doc_tokens": by[i]["tokens"]}, pre + G.question(t), t.get("max_tokens", 48))
        f.write(json.dumps({"probe": "ret", "health": "after2", "values": health(), "at": round(time.time(), 3)}) + "\n")
    return 1 if fails else 0


def warm(out, m):
    os.environ.setdefault("LCQ_DEPTHS", "131072 262144")
    os.environ.setdefault("LCQ_HAYS", "1")
    os.environ.setdefault("LCQ_WARM_TASKS", "4")
    os.environ["LCQ_SAMPLE_REPS"] = "0"
    L.LABEL = LABEL
    return L.run(out, m)


def recs(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2


def fmt(x, d=0):
    return "?" if x is None else f"{x:,.{d}f}"


def report(path):
    rs = recs(path)
    errs = sum(1 for r in rs if r.get("error"))
    if path.endswith((".warm.jsonl", ".lcqw.jsonl")):
        w = [r for r in rs if r.get("phase") == "warm" and (r.get("prompt_tokens") or 0) >= 100000 and not r.get("error")]
        ok = [r for r in w if (r.get("cached_tokens") or 0) >= 0.9 * (r.get("prompt_tokens") or 1)]
        cold = [r for r in rs if r.get("phase") == "cold" and not r.get("error")]
        return (f"same document, new question: {len(ok)} of {len(w)} warm requests at 100k+ resumed (cached >= 90%); warm first token "
                f"median {fmt(med([r.get('ttft_s') for r in w]), 2)} s, cold {fmt(med([r.get('ttft_s') for r in cold]), 2)} s; {errs} errors")
    if path.endswith(".fork.jsonl"):
        w = [r for r in rs if r.get("probe") == "fork" and (r.get("worker") or 0) >= 3 and not r.get("error")]
        ok = [r for r in w if (r.get("cached_tokens") or 0) >= 0.9 * (r.get("head_tokens_est") or 1)]
        return (f"shared opening, parted early: {len(ok)} of {len(w)} workers 3+ resumed the opening; cached median "
                f"{fmt(med([r.get('cached_tokens') for r in w]), 0)} tokens; first token median {fmt(med([r.get('ttft_s') for r in w]), 2)} s; {errs} errors")
    p2 = [r for r in rs if r.get("pass") == 2 and not r.get("error")]
    ok = [r for r in p2 if (r.get("cached_tokens") or 0) >= 0.9 * (r.get("doc_tokens") or 1)]
    read = sum(r.get("prompt_tokens") or 0 for r in rs if r.get("pass") == 1 and not r.get("error"))
    return (f"cache retention: {len(ok)} of {len(p2)} oldest documents came back warm after {read:,} tokens read; first token median "
            f"{fmt(med([r.get('ttft_s') for r in p2]), 2)} s; {errs} errors")


def main(argv):
    global LABEL
    if len(argv) >= 3 and argv[1] == "report":
        for p in argv[2:]:
            print(f"{p}: {report(p)}")
        return 0
    rest, i = [], 0
    while i < len(argv):
        if argv[i] in ("--url", "--model", "--label") and i + 1 < len(argv):
            if argv[i] == "--url":
                L.URL = argv[i + 1].rstrip("/")
            elif argv[i] == "--model":
                L.MODEL = argv[i + 1]
            else:
                LABEL = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    if len(rest) != 3 or rest[1] not in ("warm", "fork", "ret") or not L.URL or not L.MODEL:
        print(USAGE, file=sys.stderr)
        return 2
    m = G.load(os.environ.get("LCQ_MANIFEST") or G.MANIFEST)
    by = {i["id"]: i for i in m["items"]}
    try:
        if rest[1] == "warm":
            return warm(rest[2], m)
        return fork(rest[2], by) if rest[1] == "fork" else ret(rest[2], by)
    except KeyError as e:
        print(f"reuseprobe: no item {e} in the manifest", file=sys.stderr)
        return 2
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
