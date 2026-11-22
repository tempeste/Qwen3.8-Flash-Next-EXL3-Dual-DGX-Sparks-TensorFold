#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.request

USAGE = """lcq.py: runs LCQ v2 (lcqgen.py, manifest.json: the planned haystacks a depth) against an OpenAI-compatible
endpoint, one request at a time.

  lcq.py run OUT.jsonl --url URL --model MODEL [--label LABEL]
                            accuracy and task time by depth, every planned haystack

Every request is streamed (stream_options include_usage) to URL/v1/chat/completions, one user message holding the
haystack and then the question, chat_template_kwargs enable_thinking false. A record a request goes to OUT.jsonl:
phase, item, hay, depth, task, prompt and cached tokens (usage.prompt_tokens_details.cached_tokens), ttft_s, total_s,
completion tokens, finish, the engine's prefill seconds when it reports them, score, exact, temperature, seed, rep and
the reply's first 400 characters.

run, depths ascending (LCQ_DEPTHS, default all nine), haystacks h<depth>-0.. in order (LCQ_HAYS caps how many a depth):
  cold   the haystack's lookup look0, its first request (a hashed session line opens every haystack: a unique prefix);
  warm   the other 15 tasks (LCQ_WARM_TASKS: the first N only), temperature 0, top_p 1: the same prefix again, where
         the engine can reuse it, then a new question;
  samp   at depths in LCQ_SAMPLE_DEPTHS (200000 400000) on the first LCQ_SAMPLE_HAYS (2) haystacks, right after their
         warm tasks while the prefix is still cached: all 16 tasks LCQ_SAMPLE_REPS (2) times at LCQ_SAMPLE_TEMP (0.7),
         top_p 1, each request seeded (seed field) by sha256 of label, item, task and rep.
LCQ_DEADLINE (epoch seconds): the first haystack always runs; each later one starts only when the time the last
haystack of its depth took (or the deepest one run so far, scaled by depth) still fits before the deadline. Each
haystack not started is one 'unreached' record with the reason.

Environment (the options win): LCQ_URL, LCQ_MODEL, LCQ_LABEL, LCQ_DEPTHS, LCQ_HAYS, LCQ_SAMPLE_DEPTHS, LCQ_SAMPLE_HAYS,
LCQ_SAMPLE_REPS, LCQ_SAMPLE_TEMP, LCQ_WARM_TASKS, LCQ_DEADLINE, LCQ_TIMEOUT (1800 s a request), LCQ_MANIFEST, LCQ_RESUME.
LCQ_RESUME=1: OUT.jsonl's complete haystacks (look0 cold, the other warm tasks and every planned sampled request, none
with an error) are kept and skipped; partial haystacks, failed requests and 'unreached' lines are dropped and their
haystacks run again; the earlier file is kept as OUT.jsonl.before-<epoch>. Exit 0 when every request answered and
every haystack ran, 3 when the deadline left haystacks unreached, 1 when a request failed, 2 on bad arguments."""

HERE = os.path.dirname(os.path.realpath(__file__))
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("lcqgen", os.path.join(HERE, "lcqgen.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

URL = (os.environ.get("LCQ_URL") or "").rstrip("/")
MODEL = os.environ.get("LCQ_MODEL") or ""
LABEL = os.environ.get("LCQ_LABEL") or ""
TIMEOUT = float((os.environ.get("LCQ_TIMEOUT") or "1800"))
lock = threading.Lock()
fails = [0, 0]


def ints(name, default):
    v = os.environ.get(name, "").split()
    return [int(x) for x in v] if v else default


def ask(content, max_tokens, think, temperature=0.0, seed=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 1, "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": bool(think)}}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = {"text": "", "reasoning": "", "usage": None, "finish": None, "prefill_s": None}
    t0 = time.time()
    first = None
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except ValueError:
                continue
            if d.get("usage"):
                out["usage"] = d["usage"]
            tf = d.get("tensorfold") or {}
            if tf.get("prefill_seconds") is not None:
                out["prefill_s"] = tf["prefill_seconds"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                c = dl.get("content") or ""
                rc = dl.get("reasoning_content") or dl.get("reasoning") or ""
                if (c or rc) and first is None:
                    first = time.time()
                out["text"] += c
                out["reasoning"] += rc
                if ch.get("finish_reason"):
                    out["finish"] = ch["finish_reason"]
    t1 = time.time()
    out["ttft_s"] = round((first or t1) - t0, 4)
    out["total_s"] = round(t1 - t0, 4)
    return out


def one(f, item, prefix, t, phase, think=False, conc=1, max_tokens=None, temperature=0.0, rep=0):
    q = G.question(t)
    seed = int(hashlib.sha256(f"{LABEL} {item['id']} {t.get('task')} {rep}".encode()).hexdigest()[:8], 16) if temperature > 0 else None
    rec = {"label": LABEL, "phase": phase, "item": item["id"], "hay": item.get("idx"), "temperature": temperature, "seed": seed, "rep": rep, "depth": item["depth"], "task": t.get("task", "ok"), "kind": t.get("kind", "pf"),
           "pos": t.get("pos"), "think": bool(think), "conc": conc, "plan_tokens": item["tokens"] + item["q_tokens"].get(t.get("task", "ok"), 0),
           "at": round(time.time(), 3)}
    try:
        o = ask(prefix + q, max_tokens or t.get("max_tokens", 1), think, temperature, seed)
        u = o["usage"] or {}
        sc, ex, det = G.score(t, o["text"]) if t.get("kind") else (None, None, {})
        rec.update(prompt_tokens=u.get("prompt_tokens"), cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   completion_tokens=u.get("completion_tokens"), ttft_s=o["ttft_s"], total_s=o["total_s"], prefill_s=o["prefill_s"],
                   finish=o["finish"], score=sc, exact=ex, detail=det, text=o["text"][:400], reasoning_chars=len(o["reasoning"]))
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}"[:300], score=0.0 if t.get("kind") else None, exact=False if t.get("kind") else None)
    with lock:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        if rec.get("error"):
            fails[0] += 1
            fails[1] += 1
        else:
            fails[1] = 0
        print(f"{rec['phase']:7s} {rec['item']:16s} {rec['task']:6s} tokens {rec.get('prompt_tokens')} cached {rec.get('cached_tokens')} "
              f"ttft {rec.get('ttft_s')} total {rec.get('total_s')} score {rec.get('score')}{' ' + rec['error'] if rec.get('error') else ''}", file=sys.stderr, flush=True)
    if fails[1] >= 4:
        raise SystemExit("lcq: 4 requests in a row failed; the endpoint is gone")
    return rec


def items_of(m):
    return {i["id"]: i for i in m["items"]}


def resume(out, plan, sdep, shays, sreps, took):
    if not os.path.exists(out):
        print("resume: no earlier records, a first sitting", file=sys.stderr, flush=True)
        return set()
    recs = []
    with open(out) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    by = {}
    for r in recs:
        by.setdefault(r.get("item"), []).append(r)
    done, keep, part = set(), [], []
    for it in plan:
        rs = [r for r in by.get(it["id"], []) if r.get("phase") in ("cold", "warm", "samp")]
        n = len(it["q_tokens"])
        samp = sreps if it["depth"] in sdep and it["idx"] < shays else 0
        t0 = {(r["phase"] == "cold", r.get("task")) for r in rs if r.get("phase") in ("cold", "warm") and not r.get("error")}
        sp = {(r.get("rep"), r.get("task")) for r in rs if r.get("phase") == "samp" and not r.get("error")}
        if len({t for _, t in t0}) == n and sum(1 for c, _ in t0 if c) == 1 and len(sp) == samp * n:
            done.add(it["id"])
            keep.extend(rs)
            took[it["depth"]] = max(took.get(it["depth"], 0.0), sum(float(r.get("total_s") or 0) for r in rs if r.get("phase") in ("cold", "warm")))
        elif rs:
            part.append(it["id"])
    planned = {it["id"] for it in plan}
    keep.extend(r for r in recs if r.get("item") not in planned and r.get("phase") != "unreached")
    dropped = len(recs) - len(keep)
    bak = f"{out}.before-{int(time.time())}"
    os.replace(out, bak)
    with open(out, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    print(f"resume: {len(done)} haystacks already complete ({len(keep)} records kept); {len(part)} partial haystacks redone "
          f"({', '.join(part) or 'none'}); {dropped} records dropped (partial haystacks, errors and 'unreached' lines); "
          f"the earlier file is {os.path.basename(bak)}", file=sys.stderr, flush=True)
    return done


def run(out, m):
    by = items_of(m)
    depths = ints("LCQ_DEPTHS", G.DEPTHS)
    cap = int((os.environ.get("LCQ_HAYS") or "0") or 0)
    sdep = ints("LCQ_SAMPLE_DEPTHS", [200000, 400000])
    shays = int((os.environ.get("LCQ_SAMPLE_HAYS") or "2"))
    sreps = int((os.environ.get("LCQ_SAMPLE_REPS") or "2"))
    stemp = float((os.environ.get("LCQ_SAMPLE_TEMP") or "0.7"))
    deadline = float((os.environ.get("LCQ_DEADLINE") or "0") or 0)
    plan = []
    for d in depths:
        hs = sorted((i for i in by.values() if i["kind"] == "hay" and i["depth"] == d), key=lambda i: i["idx"])
        if not hs:
            print(f"lcq: depth {d} is not planned ({G.DEPTHS})", file=sys.stderr)
            return 2
        plan.extend(hs[:cap] if cap > 0 else hs)
    took = {}
    unreached = 0
    done = set()
    if os.environ.get("LCQ_RESUME") == "1":
        done = resume(out, plan, sdep, shays, sreps, took)
    with open(out, "a") as f:
        for it in plan:
            d = it["depth"]
            samp = sreps if d in sdep and it["idx"] < shays else 0
            if it["id"] in done:
                continue
            per = took.get(d)
            if per is None and took:
                k = max(took)
                per = took[k] * d / k
            need = (per or 0.0) * (1 + samp * 16 / 15 * 0.08)
            if deadline and took and time.time() + need > deadline:
                rec = {"label": LABEL, "phase": "unreached", "item": it["id"], "hay": it["idx"], "depth": d, "task": "-", "kind": "-",
                       "why": f"{int(deadline - time.time())} s left, this haystack needs about {int(need)} s", "at": round(time.time(), 3)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                unreached += 1
                print(f"unreached {it['id']}: {rec['why']}", file=sys.stderr, flush=True)
                continue
            t0 = time.time()
            prefix, tasks = G.build(it)
            one(f, it, prefix, tasks[0], "cold")
            wcap = int((os.environ.get("LCQ_WARM_TASKS") or "0") or 0)
            for t in (tasks[1:1 + wcap] if wcap > 0 else tasks[1:]):
                one(f, it, prefix, t, "warm")
            took[d] = time.time() - t0
            for r in range(1, samp + 1):
                for t in tasks:
                    one(f, it, prefix, t, "samp", temperature=stemp, rep=r)
    if fails[0]:
        return 1
    return 3 if unreached else 0


def endpoint(argv):
    global URL, MODEL, LABEL
    rest, i = [], 0
    while i < len(argv):
        if argv[i] in ("--url", "--model", "--label") and i + 1 < len(argv):
            v = argv[i + 1]
            if argv[i] == "--url":
                URL = v.rstrip("/")
            elif argv[i] == "--model":
                MODEL = v
            else:
                LABEL = v
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return rest


def main(argv):
    argv = endpoint(argv)
    if len(argv) != 3 or argv[1] != "run" or not URL or not MODEL:
        print(USAGE, file=sys.stderr)
        return 2
    m = G.load(os.environ.get("LCQ_MANIFEST") or G.MANIFEST)
    try:
        return run(argv[2], m)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
