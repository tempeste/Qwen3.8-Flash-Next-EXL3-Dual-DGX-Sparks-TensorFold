#!/usr/bin/env python3
import hashlib
import http.client
import importlib.util
import json
import math
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone

USAGE = """agentload.py: a synthetic agent workload against OpenAI-compatible endpoints: S streams, each holding a long
conversation built from LCQ v2's filler (lcqgen.py, corpus/corpus.jsonl.gz) and asking for a function every turn,
while a share of the streams reads a fresh long context cold. It is generated from a seed, not replayed traffic.

  agentload.py run OUT.jsonl --url URL [--url URL2 ...] --model MODEL [options]
  agentload.py plan [options]                 the turn sequence, its sha256, sizes and KV demand (no requests)
  agentload.py show STREAM TURN [options]     print one planned request body (earlier replies as fixed text)
  agentload.py verify [options]               corpus and LCQ checks, token estimate against the manifest, plan twice
  agentload.py report OUT.jsonl [...]         one block a file; medians and 95% intervals across runs of one plan
  agentload.py compare A.jsonl [...] -- B.jsonl [...]          A against B, same work required
  agentload.py table --arm NAME FILE [...] --arm NAME FILE [...]   arms side by side, same work required

A stream is pinned to URL number (stream mod URLs). Its turn 0 opens a conversation: a system line, then a context
of about CTX tokens (an LCQ v2 style haystack with its own seeded header, block order and needles, so no other
conversation shares its prefix), then a coding task. A later turn is cold (a new conversation, CTX drawn again)
with probability COLD_SHARE, or when the next warm turn would not fit MAX_CONTEXT; otherwise it is warm: a tool
result of CHUNK tokens (corpus blocks) and the next task are appended to the conversation, after the previous reply,
so the prompt starts with everything the server read and wrote last turn. Every draw comes from splitmix64 keyed by
(seed, salt, stream): stream k does the same turns whatever S is, so the same seed is the same work everywhere.
Token counts are estimates (corpus blocks counted with the model's tokenizer, other text at 3.3 characters a
token, a few tokens high), and a conversation's budget counts every earlier reply as MAX_TOKENS long.

Options (environment in brackets; the options win):
  --url URL           repeat for several servers [AL_URLS, space separated]
  --model NAME        [AL_MODEL]               --label TEXT [AL_LABEL]
  --streams S         16 [AL_STREAMS]
  --ctx LIST          context sizes a new conversation draws from, tokens; 100000,131072,200000,240000 [AL_CTX]
  --min-context N     100000 [AL_MIN_CONTEXT]; a size below it is refused unless --allow-short [AL_ALLOW_SHORT=1]
  --max-context N     262144 [AL_MAX_CONTEXT]: prompt plus reply never passes it; above 262144 (to 409600, sizes
                      to 400000) only with --long [AL_LONG=1]
  --chunk LO-HI       2048-8192 [AL_CHUNK]     tool-result tokens a warm turn appends
  --cold-share P      0.02 [AL_COLD_SHARE]    share of turns after the first that open a new conversation by draw
                      (a conversation that fills MAX_CONTEXT also starts again; plan prints both)
  --max-tokens N      1024 [AL_MAX_TOKENS]     --temperature T 0 [AL_TEMPERATURE]   --thinking on|off off [AL_THINKING]
  --fixed-length      send ignore_eos and min_tokens = max_tokens (vLLM and SGLang extensions) [AL_FIXED_LENGTH=1]
  --reply real|fixed  real [AL_REPLY]: append the model's reply; fixed: append a fixed line, so every byte is planned
  --warmup S          120 [AL_WARMUP]          --measure S 600 [AL_MEASURE]        --drain S 120 [AL_DRAIN]
  --warmup-max S      1800 [AL_WARMUP_MAX]: measuring starts at the later of WARMUP and the end of every stream's
                      first turn (its cold read of CTX tokens), but no later than WARMUP_MAX; 0: at WARMUP
  --stagger S         60 [AL_STAGGER]: stream k starts at k*STAGGER/S
  --seed N            8181 [AL_SEED]           --salt TEXT '' [AL_SALT]: new headers, same sizes and turn kinds
  --turns N           0 [AL_TURNS]: stop a stream after N turns (0: run for the duration)
  --slots N           0 [AL_SLOTS]: a URL's slots, to count requests sent while all were busy (slot waits)
  --kv-tokens N       0 [AL_KV_TOKENS]: a URL's KV pool in tokens, for plan's fit estimate
  --retry-wait S      2 [AL_RETRY_WAIT]: wait before resending a refused turn (no KV room, busy)
  --margin N          1024 [AL_MARGIN]: tokens kept free under MAX_CONTEXT for the estimate's error
Environment only: AL_TIMEOUT (1800 s a read), AL_API_KEY (sent as a bearer token), AL_HORIZON (64 turns a stream in
the plan's sha256), AL_SLOW_FIRST (10 s: warm first tokens slower than this are counted), AL_FORCE=1 (compare and
table across different work), AL_BENCH_DIR (where lcqgen.py lives; default this script's folder).

Measured for MEASURE seconds from the window start above (DRAIN more seconds for replies in flight, then they are
cut). The same seed sends the same bytes, so restart the server between runs or its prompt cache holds the last run. Refused turns are resent after RETRY_WAIT and every attempt is recorded. Exit 0 when every turn answered,
1 when a request failed or was refused, 2 on bad arguments."""

HERE = os.path.dirname(os.path.realpath(__file__))
BENCH = os.environ.get("AL_BENCH_DIR") or HERE
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("lcqgen", os.path.join(BENCH, "lcqgen.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

VERSION = 1
TASKS_VERSION = "agentload-tasks-1"
CPT = 3.3
MSG = 8
LONG_CONTEXT = 262144
LONG_MAX = 409600
LONG_CTX = 400000
REFUSED = ("no_room", "busy")
SYSTEM = ("You are a coding agent in a long work session. The conversation holds the tool results and notes gathered "
          "so far. Answer each task with one fenced code block, then at most two sentences.")
PREAMBLE = "Work session, stream {s}, conversation {c}. Tool results and notes so far:\n\n"
CHUNK_HEAD = "Tool results since your last reply:\n\n"

TASKS = [
    ("T01", "Write a Python function parse_duration(s: str) -> int that turns strings like '1h30m', '45s' or '2h' into seconds and raises ValueError on anything else."),
    ("T02", "Write a Python function merge_intervals(xs: list[tuple[int, int]]) -> list[tuple[int, int]] that merges overlapping or touching closed intervals and returns them sorted."),
    ("T03", "Write a Python function top_k_words(text: str, k: int) -> list[tuple[str, int]] that returns the k most frequent lower-cased words, ties broken alphabetically."),
    ("T04", "Write a Python function is_valid_ipv4(s: str) -> bool that accepts dotted quads 0-255 without leading zeros."),
    ("T05", "Write a Python function chunk(xs: list, n: int) -> list[list] that splits a list into consecutive pieces of n items, the last one shorter, and raises ValueError when n < 1."),
    ("T06", "Write a Python function lru_get(cache: dict, order: list, key, capacity: int, load) that returns the cached value for key, loading and evicting the least recently used entry as needed."),
    ("T07", "Write a Python function roman_to_int(s: str) -> int for Roman numerals up to 3999, rejecting malformed input."),
    ("T08", "Write a Python function flatten(d: dict, sep: str = '.') -> dict that flattens nested dictionaries into dotted keys."),
    ("T09", "Write a Python function diff_lines(a: str, b: str) -> list[str] that returns a minimal line diff with '+ ', '- ' and '  ' prefixes."),
    ("T10", "Write a Python function retry(fn, attempts: int, base_delay: float) that calls fn with exponential backoff and re-raises the last error."),
    ("T11", "Write a Python function semver_compare(a: str, b: str) -> int that compares versions like '1.4.0-rc.2' by the SemVer 2.0 rules."),
    ("T12", "Write a Python function topo_sort(deps: dict[str, list[str]]) -> list[str] that orders nodes so each comes after its dependencies and raises ValueError on a cycle."),
    ("T13", "Write a Python function parse_csv_line(line: str) -> list[str] that handles quoted fields, doubled quotes and commas inside quotes."),
    ("T14", "Write a Python function rolling_mean(xs: list[float], w: int) -> list[float] that returns the mean of every window of w consecutive values in O(n)."),
    ("T15", "Write a Python function slugify(title: str) -> str that lower-cases, keeps ASCII letters and digits, and joins words with single hyphens."),
    ("T16", "Write a Python function bucket_by_hour(events: list[tuple[int, str]]) -> dict[int, int] that counts events per UTC hour from epoch seconds."),
    ("T17", "Write a Python function longest_common_prefix(xs: list[str]) -> str."),
    ("T18", "Write a Python function balanced(s: str) -> bool that checks (), [] and {} nesting and ignores every other character."),
    ("T19", "Write a Python function human_bytes(n: int) -> str that prints sizes like '1.5 KiB', '20 MiB' with one decimal below 10."),
    ("T20", "Write a Python function dedupe_keep_order(xs: list) -> list that drops repeats and keeps first occurrences, for unhashable items too."),
    ("T21", "Write a Python function parse_query(qs: str) -> dict[str, list[str]] that decodes a URL query string with repeated keys and percent escapes."),
    ("T22", "Write a Python function binary_search_left(xs: list[int], x: int) -> int that returns the first index whose value is >= x."),
    ("T23", "Write a Python function wrap_text(s: str, width: int) -> list[str] that wraps words greedily and splits words longer than width."),
    ("T24", "Write a Python function cron_matches(expr: str, minute: int, hour: int) -> bool for the minute and hour fields of a cron expression with *, lists, ranges and steps."),
    ("T25", "Write a Python function group_anagrams(words: list[str]) -> list[list[str]] that groups words that are anagrams, each group sorted, groups in first-seen order."),
    ("T26", "Write a Python function json_path_get(obj, path: str, default=None) that reads paths like 'a.b[2].c' from nested dicts and lists."),
    ("T27", "Write a Python function rate_limiter(capacity: int, refill_per_s: float) that returns an allow(now: float) -> bool token-bucket closure."),
    ("T28", "Write a Python function levenshtein(a: str, b: str) -> int using two rows of memory."),
    ("T29", "Write a Python function parse_ini(text: str) -> dict[str, dict[str, str]] that reads sections, key = value lines and ignores lines starting with ';' or '#'."),
    ("T30", "Write a Python function percentile(xs: list[float], p: float) -> float using the nearest-rank method and raising ValueError on an empty list."),
    ("T31", "Write a Python function shortest_path(grid: list[str]) -> int that returns the BFS step count from 'S' to 'E' through '.' cells, or -1."),
    ("T32", "Write a Python function format_table(rows: list[list[str]]) -> str that left-aligns columns padded to their widest cell."),
    ("T33", "Write a Python function split_sentences(text: str) -> list[str] that splits on '.', '!' and '?' followed by whitespace, keeping common abbreviations like 'e.g.' intact."),
    ("T34", "Write a Python function crc8(data: bytes) -> int with polynomial 0x07, initial value 0 and no reflection."),
    ("T35", "Write a Python function interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float that returns the length both half-open intervals share."),
    ("T36", "Write a Python function camel_to_snake(name: str) -> str that handles acronyms, so 'HTTPServerError' becomes 'http_server_error'."),
]

OPTS = {
    "--url": ("AL_URLS", "", "urls"), "--model": ("AL_MODEL", "", "str"), "--label": ("AL_LABEL", "", "str"),
    "--streams": ("AL_STREAMS", "16", "int"), "--ctx": ("AL_CTX", "100000,131072,200000,240000", "ints"),
    "--min-context": ("AL_MIN_CONTEXT", "100000", "int"), "--allow-short": ("AL_ALLOW_SHORT", "", "flag"),
    "--max-context": ("AL_MAX_CONTEXT", "262144", "int"), "--long": ("AL_LONG", "", "flag"),
    "--chunk": ("AL_CHUNK", "2048-8192", "range"), "--cold-share": ("AL_COLD_SHARE", "0.02", "float"),
    "--max-tokens": ("AL_MAX_TOKENS", "1024", "int"), "--temperature": ("AL_TEMPERATURE", "0", "float"),
    "--thinking": ("AL_THINKING", "off", "onoff"), "--fixed-length": ("AL_FIXED_LENGTH", "", "flag"),
    "--reply": ("AL_REPLY", "real", "str"), "--warmup": ("AL_WARMUP", "120", "float"), "--warmup-max": ("AL_WARMUP_MAX", "1800", "float"),
    "--measure": ("AL_MEASURE", "600", "float"), "--drain": ("AL_DRAIN", "120", "float"),
    "--stagger": ("AL_STAGGER", "60", "float"), "--seed": ("AL_SEED", "8181", "int"), "--salt": ("AL_SALT", "", "str"),
    "--turns": ("AL_TURNS", "0", "int"), "--slots": ("AL_SLOTS", "0", "int"), "--kv-tokens": ("AL_KV_TOKENS", "0", "int"),
    "--retry-wait": ("AL_RETRY_WAIT", "2", "float"), "--margin": ("AL_MARGIN", "1024", "int"),
}


class ArgError(Exception):
    pass


def conv(kind, v, name):
    try:
        if kind == "urls":
            return [u.rstrip("/") for u in v.split()]
        if kind == "int":
            return int(v)
        if kind == "float":
            return float(v)
        if kind == "ints":
            return [int(x) for x in v.replace(",", " ").split()]
        if kind == "range":
            a, _, b = v.partition("-")
            lo, hi = int(a), int(b or a)
            if lo < 1 or hi < lo:
                raise ValueError
            return (lo, hi)
        if kind == "flag":
            return v not in ("", "0", "false", "no", "off")
        if kind == "onoff":
            if v not in ("on", "off"):
                raise ValueError
            return v == "on"
        return v
    except ValueError:
        raise ArgError(f"{name}: bad value {v!r}")


def parse(argv):
    o, pos, urls, i = {}, [], [], 0
    for f, (env, d, kind) in OPTS.items():
        o[f[2:].replace("-", "_")] = conv(kind, os.environ.get(env, d), env)
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            raise ArgError("")
        if a in OPTS:
            kind = OPTS[a][2]
            if kind == "flag":
                o[a[2:].replace("-", "_")] = True
                i += 1
                continue
            if i + 1 >= len(argv):
                raise ArgError(f"{a} needs a value")
            if kind == "urls":
                urls.append(argv[i + 1].rstrip("/"))
            else:
                o[a[2:].replace("-", "_")] = conv(kind, argv[i + 1], a)
            i += 2
        elif a.startswith("--"):
            raise ArgError(f"unknown option {a}")
        else:
            pos.append(a)
            i += 1
    if urls:
        o["url"] = urls
    check(o)
    return pos, o


def check(o):
    if o["reply"] not in ("real", "fixed"):
        raise ArgError("--reply is real or fixed")
    if o["streams"] < 1:
        raise ArgError("--streams must be at least 1")
    if not 0 <= o["cold_share"] <= 1:
        raise ArgError("--cold-share is between 0 and 1")
    if not o["ctx"]:
        raise ArgError("--ctx needs at least one size")
    short = [n for n in o["ctx"] if n < o["min_context"]]
    if short and not o["allow_short"]:
        raise ArgError(f"--ctx {','.join(map(str, short))} is below --min-context {o['min_context']}: a held conversation "
                       f"under it does not exercise long-context serving; pass --allow-short to run it anyway")
    cap = LONG_MAX if o["long"] else LONG_CONTEXT
    big = LONG_CTX if o["long"] else LONG_CONTEXT
    if o["max_context"] > cap:
        raise ArgError(f"--max-context {o['max_context']} is above {cap}" + ("" if o["long"] else
                       " (262,144-token windows); pass --long for up to 409,600"))
    over = [n for n in o["ctx"] if n > big]
    if over:
        raise ArgError(f"--ctx {','.join(map(str, over))} is above {big}" + ("" if o["long"] else "; pass --long for up to 400,000"))
    nofit = [n for n in o["ctx"] if n + o["max_tokens"] + o["margin"] > o["max_context"]]
    if nofit:
        raise ArgError(f"--ctx {','.join(map(str, nofit))}: a first turn of that size plus --max-tokens {o['max_tokens']} and "
                       f"--margin {o['margin']} does not fit --max-context {o['max_context']}")


def est_text(s):
    return int(math.ceil(len(s) / CPT))


def task_prompt(tid):
    return f"Task {tid} ({TASKS_VERSION}): {dict(TASKS)[tid]}"


def work_settings(o):
    keys = ("streams", "ctx", "min_context", "max_context", "chunk", "cold_share", "max_tokens", "temperature", "thinking",
            "fixed_length", "reply", "seed", "salt", "margin")
    return {k: (list(o[k]) if isinstance(o[k], tuple) else o[k]) for k in keys}


class Plan:
    def __init__(self, o):
        self.o = o
        self.c = G.corpus()
        self.N = len(self.c)
        self.kind = "agentload" + (f"-{o['salt']}" if o["salt"] else "")
        self.lock = threading.Lock()
        self.st = {}
        self.task_max = max(est_text(task_prompt(t)) for t, _ in TASKS)
        self.head_est = est_text(SYSTEM) + MSG + est_text(PREAMBLE.format(s=99999, c=99999)) + MSG

    def perm(self, *key):
        o = list(range(self.N))
        r = G.R(*key)
        for i in range(self.N - 1, 0, -1):
            j = r.below(i + 1)
            o[i], o[j] = o[j], o[i]
        return o

    def stream(self, s):
        if s not in self.st:
            o = self.o
            r = G.R(o["seed"], o["salt"], "agentload", s)
            tasks = [t for t, _ in TASKS]
            for i in range(len(tasks) - 1, 0, -1):
                j = r.below(i + 1)
                tasks[i], tasks[j] = tasks[j], tasks[i]
            self.st[s] = {"r": r, "tasks": tasks, "turns": [], "conv": -1, "est": 0, "cur": 0,
                          "perm": self.perm(o["seed"], o["salt"], "agentload-chunks", s)}
        return self.st[s]

    def turn(self, s, t):
        with self.lock:
            st = self.stream(s)
            while len(st["turns"]) <= t:
                st["turns"].append(self.next(s, st))
            return st["turns"][t]

    def chunk(self, st, target):
        out, n, k = [], 0, st["cur"]
        while n < target or not out:
            b = st["perm"][k % self.N]
            out.append(b)
            n += self.c[b]["n"]
            k += 1
        return out, n, k

    def context(self, s, conv, size):
        o = self.o
        it = {"seed": o["seed"], "kind": self.kind, "depth": size, "idx": s * 100000 + conv,
              "start": G.R(o["seed"], o["salt"], "agentload-ctx", s, conv).below(self.N), "nblocks": 0, "tail": 0}
        key = (it["seed"], it["kind"], it["depth"], it["idx"])
        order = G.order(it)
        G._orders.pop(key, None)
        ins, _ = G.needles(it["seed"], it["kind"], size, it["idx"])
        other = len(G.header(it["seed"], it["kind"], size, it["idx"])) + sum(len(t) for _, t in ins)
        budget = size - self.head_est - self.task_max - MSG
        acc, nb, bchars = est_text("x" * other), 0, 0
        while acc + self.c[order[(it["start"] + nb) % self.N]]["n"] <= budget:
            b = self.c[order[(it["start"] + nb) % self.N]]
            acc += b["n"]
            bchars += len(b["t"])
            nb += 1
        tail_block = self.c[order[(it["start"] + nb) % self.N]]
        room, lines, used = (budget - acc) * CPT, 0, 0
        for ln in tail_block["t"].rstrip("\n").split("\n"):
            if used + len(ln) + 4 + 4 > room:
                break
            used += len(ln) + 1
            lines += 1
        it["nblocks"], it["tail"] = nb, lines
        tail = len(G.tail_text(tail_block, lines))
        est = (acc - est_text("x" * other)) + est_text("x" * (other + tail))
        return it, est

    def next(self, s, st):
        o = self.o
        t = len(st["turns"])
        r = st["r"]
        u = r.below(1000000) / 1000000.0
        size = o["ctx"][r.below(len(o["ctx"]))]
        lo, hi = o["chunk"]
        target = lo + r.below(hi - lo + 1)
        tid = st["tasks"][t % len(st["tasks"])]
        task_est = est_text(task_prompt(tid)) + 1
        cold = t == 0 or u < o["cold_share"]
        forced = False
        if not cold:
            blocks, n, k = self.chunk(st, target)
            est = st["est"] + MSG + est_text(CHUNK_HEAD) + n + task_est + MSG
            if est + o["max_tokens"] + o["margin"] > o["max_context"]:
                cold, forced = True, True
            else:
                st["cur"] = k
                d = {"t": t, "cold": False, "conv": st["conv"], "blocks": blocks, "chunk_tokens": n, "task": tid, "est": est}
        if cold:
            st["conv"] += 1
            it, cest = self.context(s, st["conv"], size)
            est = self.head_est + cest + task_est
            d = {"t": t, "cold": True, "forced": forced, "conv": st["conv"], "ctx": size, "item": it, "task": tid, "est": est}
        st["est"] = d["est"] + o["max_tokens"]
        return d

    def text(self, s, d):
        if d["cold"]:
            return PREAMBLE.format(s=s, c=d["conv"]) + G.build(d["item"])[0] + "\n" + task_prompt(d["task"])
        return CHUNK_HEAD + "".join(self.c[b]["t"] for b in d["blocks"]) + "\n" + task_prompt(d["task"])

    def sha(self, horizon):
        h = hashlib.sha256()
        h.update(json.dumps({"v": VERSION, "tasks": TASKS_VERSION, "tasks_sha": hashlib.sha256(json.dumps(TASKS).encode()).hexdigest(),
                             "system": SYSTEM, "work": work_settings(self.o)}, sort_keys=True).encode())
        for s in range(self.o["streams"]):
            for t in range(horizon):
                h.update(json.dumps(self.turn(s, t), sort_keys=True).encode())
        return h.hexdigest()


def fixed_reply(tid):
    return f"(reply to task {tid} left out of this conversation)"


def body(o, messages):
    b = {"model": o["model"], "messages": messages, "max_tokens": o["max_tokens"], "temperature": o["temperature"], "top_p": 1,
         "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": bool(o["thinking"])}}
    if o["temperature"] > 0:
        b["seed"] = int(hashlib.sha256(json.dumps([m["content"][-64:] for m in messages]).encode()).hexdigest()[:8], 16)
    if o["fixed_length"]:
        b["ignore_eos"] = True
        b["min_tokens"] = o["max_tokens"]
    return json.dumps(b).encode()


def planned_messages(plan, s, t):
    hist = []
    for k in range(t + 1):
        d = plan.turn(s, k)
        if d["cold"]:
            hist = [{"role": "system", "content": SYSTEM}]
        hist.append({"role": "user", "content": plan.text(s, d)})
        if k < t:
            hist.append({"role": "assistant", "content": fixed_reply(d["task"])})
    return hist


NO_ROOM = re.compile(r"no ?room|kv[ _-]?cache|kv room|kv blocks|out of (kv|blocks|memory|pages)|not enough (kv|memory|blocks|room|pages)|insufficient (kv|memory|blocks)|cannot allocate|allocation fail", re.I)
TOO_LONG = re.compile(r"context length|context size|context window|maximum context|too long|exceeds the|max_model_len|longer than", re.I)
BUSY = re.compile(r"busy|no (free |idle )?slot|all slots|overloaded|too many|try again|unavailable|queue (is )?full", re.I)


def classify(status, msg):
    msg = msg or ""
    if NO_ROOM.search(msg):
        return "no_room"
    if TOO_LONG.search(msg):
        return "too_long"
    if status in (429, 503) or BUSY.search(msg):
        return "busy"
    return "error"


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Run:
    def __init__(self, o, out):
        self.o, self.out, self.plan = o, out, Plan(o)
        self.urls = o["url"]
        self.parsed = [urllib.parse.urlsplit(u) for u in self.urls]
        self.lock = threading.Lock()
        self.inflight = [0] * len(self.urls)
        self.conns = {}
        self.stop = threading.Event()
        self.aborting = False
        self.failed = 0
        self.timeout = float(os.environ.get("AL_TIMEOUT") or "1800")
        self.first_done = set()
        self.key = os.environ.get("AL_API_KEY") or ""

    def rel(self):
        return time.monotonic() - self.t0

    def write(self, rec):
        with self.lock:
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()

    def send(self, s, u, d, payload, attempt, reusable):
        p = self.parsed[u]
        cls = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
        conn = cls(p.hostname, p.port or (443 if p.scheme == "https" else 80), timeout=self.timeout)
        path = (p.path or "") + "/v1/chat/completions"
        hdr = {"Content-Type": "application/json"}
        if self.key:
            hdr["Authorization"] = f"Bearer {self.key}"
        rec = {"kind": "turn", "label": self.o["label"], "stream": s, "u": u, "turn": d["t"], "conv": d["conv"], "cold": d["cold"],
               "forced": d.get("forced", False), "ctx": d.get("ctx"), "task": d["task"], "attempt": attempt, "est_prompt": d["est"],
               "reusable": reusable, "body_sha256": hashlib.sha256(payload).hexdigest()}
        with self.lock:
            rec["inflight"] = self.inflight[u]
            self.inflight[u] += 1
            self.conns[s] = conn
        t0 = time.monotonic()
        rec["send"] = round(t0 - self.t0, 4)
        arr, text, usage, finish, first, status, msg, code = [], [], None, None, None, "ok", None, None
        try:
            conn.request("POST", path, payload, hdr)
            resp = conn.getresponse()
            code = resp.status
            if resp.status != 200:
                msg = resp.read(4000).decode("utf-8", "replace")
                status = classify(resp.status, msg)
            else:
                while True:
                    raw = resp.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    if ev.get("error"):
                        e = ev["error"]
                        msg = json.dumps(e) if not isinstance(e, str) else e
                        status = classify(0, msg)
                        break
                    if ev.get("usage"):
                        usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        dl = ch.get("delta") or {}
                        c = dl.get("content") or ""
                        rc = dl.get("reasoning_content") or dl.get("reasoning") or ""
                        if c or rc:
                            tm = time.monotonic()
                            if first is None:
                                first = tm
                            arr.append(int(round((tm - t0) * 1000)))
                        text.append(c)
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
        except Exception as e:
            status = "aborted" if self.aborting else "error"
            msg = f"{type(e).__name__}: {e}"
        finally:
            try:
                conn.close()
            except Exception:
                pass
            with self.lock:
                self.inflight[u] -= 1
                self.conns.pop(s, None)
        t1 = time.monotonic()
        rec.update(first=None if first is None else round(first - self.t0, 4), end=round(t1 - self.t0, 4), status=status, http=code,
                   finish=finish, arr=arr)
        if usage:
            rec.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                       cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        if msg:
            rec["msg"] = msg[:300]
        self.write(rec)
        if status != "ok":
            print(f"stream {s} turn {d['t']} attempt {attempt}: {status} {code or ''} {(msg or '')[:160]}", file=sys.stderr, flush=True)
        return rec, "".join(text)

    def stream(self, s):
        o = self.o
        u = s % len(self.urls)
        start = s * o["stagger"] / o["streams"]
        if self.stop.wait(max(0.0, start - self.rel())):
            return
        hist, prev, errs, t = [], None, 0, 0
        while not self.stop.is_set() and (o["turns"] <= 0 or t < o["turns"]):
            d = self.plan.turn(s, t)
            text = self.plan.text(s, d)
            if d["cold"]:
                hist, reusable = [{"role": "system", "content": SYSTEM}], None
            else:
                reusable = (prev["prompt_tokens"] + (prev.get("completion_tokens") or 0)) if prev and prev.get("prompt_tokens") else None
            hist.append({"role": "user", "content": text})
            payload = body(o, hist)
            attempt = 0
            while True:
                attempt += 1
                rec, reply = self.send(s, u, d, payload, attempt, reusable)
                if rec["status"] not in ("ok", "aborted"):
                    self.failed += 1
                if rec["status"] in REFUSED and not self.stop.wait(o["retry_wait"]):
                    continue
                break
            if t == 0 and rec["status"] not in REFUSED:
                self.first_done.add(s)
            if rec["status"] in REFUSED:
                break
            if rec["status"] == "ok":
                errs = 0
                hist.append({"role": "assistant", "content": reply if o["reply"] == "real" else fixed_reply(d["task"])})
                prev = rec
            else:
                errs += 1
                hist.append({"role": "assistant", "content": fixed_reply(d["task"])})
                prev = None
                if errs >= 4:
                    print(f"stream {s}: 4 requests in a row failed; the stream stops", file=sys.stderr, flush=True)
                    break
            t += 1

    def run(self):
        o = self.o
        horizon = int(os.environ.get("AL_HORIZON") or "64")
        sha = self.plan.sha(horizon)
        per = [sum(1 for s in range(o["streams"]) if s % len(self.urls) == u) for u in range(len(self.urls))]
        if o["slots"] and max(per) > o["slots"]:
            print(f"agentload: {max(per)} streams on a URL with {o['slots']} slots: the rest wait for a slot; "
                  f"requests sent while all slots were busy are counted as slot waits", file=sys.stderr, flush=True)
        with open(self.out, "a") as self.f:
            self.t0 = time.monotonic()
            self.write({"kind": "run", "v": VERSION, "label": o["label"], "started_utc": now_utc(), "model": o["model"],
                        "urls": self.urls, "streams_per_url": per, "work": work_settings(o), "plan_sha256": sha, "horizon": horizon,
                        "tasks_version": TASKS_VERSION, "corpus_sha256": G.load(os.environ.get("LCQ_MANIFEST") or G.MANIFEST)["corpus_sha256"],
                        "warmup": o["warmup"], "warmup_max": o["warmup_max"], "measure": o["measure"], "drain": o["drain"], "stagger": o["stagger"], "turns": o["turns"],
                        "slots": o["slots"], "kv_tokens": o["kv_tokens"], "retry_wait": o["retry_wait"]})
            th = [threading.Thread(target=self.stream, args=(s,), daemon=True) for s in range(o["streams"])]
            for x in th:
                x.start()
            cap = max(o["warmup"], o["warmup_max"])
            while any(x.is_alive() for x in th) and self.rel() < cap and (
                    self.rel() < o["warmup"] or (o["warmup_max"] > 0 and len(self.first_done) < o["streams"])):
                time.sleep(0.05)
            W = round(min(self.rel(), cap), 4)
            warm = len(self.first_done)
            self.write({"kind": "window", "W": W, "warm": warm})
            print(f"agentload: measuring from {W:.0f} s, {warm} of {o['streams']} streams past their first turn", file=sys.stderr, flush=True)
            end = W + o["measure"]
            while self.rel() < end and any(x.is_alive() for x in th):
                time.sleep(0.05)
            stop = min(self.rel(), end)
            self.stop.set()
            dl = time.monotonic() + o["drain"]
            for x in th:
                x.join(max(0.0, dl - time.monotonic()))
            with self.lock:
                self.aborting = True
                cut = list(self.conns.values())
            for c in cut:
                try:
                    c.sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
            for x in th:
                x.join(30)
            self.write({"kind": "end", "W": W, "stop": round(stop, 4), "ended": round(self.rel(), 4), "cut": len(cut), "ended_utc": now_utc()})
        return 1 if self.failed else 0


def load(path):
    hdr, end, turns = None, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            k = r.get("kind")
            if k == "run":
                hdr = r
            elif k == "end":
                end = r
            elif k == "turn":
                turns.append(r)
    if hdr is None:
        raise ValueError(f"{path}: no run line")
    return hdr, turns, end


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[max(0, int(math.ceil(p / 100.0 * len(xs))) - 1)]


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(a, b, iv):
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in iv)


def metrics(hdr, turns, end):
    W = float(end["W"]) if end and end.get("W") is not None else float(hdr["warmup"])
    stop = end["stop"] if end else max([r["end"] for r in turns] or [W])
    E = min(W + float(hdr["measure"]), stop)
    M = E - W
    nurl = len(hdr["urls"])
    slots = hdr.get("slots") or 0
    slow = float(os.environ.get("AL_SLOW_FIRST") or "10")
    m = {"window_s": round(M, 3), "window_start": W, "urls": nurl, "streams": hdr["work"]["streams"]}
    inw = lambda t: t is not None and W <= t < E
    gen = [0.0] * nurl
    newp = [0.0] * nurl
    cached = prompt = 0
    wc = wp = 0
    resumed = reuse_n = 0
    coldiv = [[] for _ in range(nurl)]
    for r in turns:
        if r["cold"] and r["status"] not in REFUSED:
            coldiv[r["u"]].append((r["send"], r["first"] if r["first"] is not None else r["end"]))
    coldiv = [merge(x) for x in coldiv]
    acc = {}
    gaps = []
    tw, tc = [], []
    no_usage = 0
    for r in turns:
        u = r["u"]
        arr = [r["send"] + a / 1000.0 for a in r.get("arr") or []]
        ct = r.get("completion_tokens")
        w = (ct / len(arr)) if ct and arr else 1.0
        gen[u] += w * sum(1 for a in arr if inw(a))
        pt = r.get("prompt_tokens")
        if pt is not None:
            ca = r.get("cached_tokens") or 0
            a, b = r["send"], r["first"] if r["first"] is not None else r["end"]
            new = pt - ca
            if b > a:
                newp[u] += new * max(0.0, min(b, E) - max(a, W)) / (b - a)
            elif inw(a):
                newp[u] += new
            if inw(b):
                cached += ca
                prompt += pt
                if not r["cold"]:
                    wc += ca
                    wp += pt
                    if r.get("reusable"):
                        reuse_n += 1
                        resumed += ca >= 0.9 * r["reusable"]
        elif r["status"] not in REFUSED and r.get("arr"):
            no_usage += 1
        if r["first"] is not None and inw(r["send"]):
            (tc if r["cold"] else tw).append(r["first"] - r["send"])
        st = acc.setdefault(r["stream"], [0.0, 0.0, 0.0, 0.0])
        for x, y in zip(arr, arr[1:]):
            if inw(y):
                gaps.append(y - x)
            a, b = max(x, W), min(y, E)
            if y > x and b > a:
                tok = w * (b - a) / (y - x)
                c = overlap(a, b, coldiv[u])
                st[0] += tok * c / (b - a)
                st[1] += c
                st[2] += tok * (b - a - c) / (b - a)
                st[3] += b - a - c
            elif y == x and inw(y):
                st[0 if overlap(y - 1e-9, y + 1e-9, coldiv[u]) > 0 else 2] += w
    m["gen_tps"] = sum(gen) / M if M > 0 else None
    m["gen_tps_url"] = [g / M if M > 0 else None for g in gen]
    m["new_prompt_tps"] = sum(newp) / M if M > 0 else None
    m["new_prompt_tps_url"] = [g / M if M > 0 else None for g in newp]
    m["cached_tokens"] = cached
    m["prompt_tokens"] = prompt
    m["cached_tps"] = cached / M if M > 0 else None
    m["ratio"] = (m["new_prompt_tps"] / m["gen_tps"]) if m["gen_tps"] else None
    m["warm_cached_share"] = wc / wp if wp else None
    m["warm_resumed"] = resumed
    m["warm_reuse_n"] = reuse_n
    m["no_usage"] = no_usage
    m["decode_cold_med"] = med([a[0] / a[1] for a in acc.values() if a[1] >= 1.0])
    m["decode_none_med"] = med([a[2] / a[3] for a in acc.values() if a[3] >= 1.0])
    tcold = sum(a[1] for a in acc.values())
    tnone = sum(a[3] for a in acc.values())
    m["decode_cold_all"] = sum(a[0] for a in acc.values()) / tcold if tcold else None
    m["decode_none_all"] = sum(a[2] for a in acc.values()) / tnone if tnone else None
    m["cold_time_share"] = tcold / (tcold + tnone) if tcold + tnone else None
    for name, xs in (("warm", tw), ("cold", tc)):
        m[f"ttft_{name}_n"] = len(xs)
        for p in (50, 90, 99):
            m[f"ttft_{name}_p{p}"] = pct(xs, p)
    m["gap_p99"] = pct(gaps, 99)
    m["gap_max"] = max(gaps) if gaps else None
    m["gaps_over_1s"] = sum(1 for g in gaps if g > 1.0)
    ok = [r for r in turns if r["status"] == "ok"]
    m["turns_done"] = sum(1 for r in ok if inw(r["end"]))
    m["cold_done"] = sum(1 for r in ok if inw(r["end"]) and r["cold"])
    m["forced_done"] = sum(1 for r in ok if inw(r["end"]) and r.get("forced"))
    m["turns_done_run"] = len(ok)
    reach = {}
    for r in ok:
        reach[r["stream"]] = max(reach.get(r["stream"], -1), r["turn"])
    rs = [reach.get(s, -1) + 1 for s in range(hdr["work"]["streams"])]
    m["reach_min"], m["reach_med"], m["reach_max"] = min(rs), med(rs), max(rs)
    m["reached_turns"] = sum(rs)
    for k in ("no_room", "busy", "too_long", "error", "aborted"):
        m[k] = sum(1 for r in turns if r["status"] == k)
        m[k + "_window"] = sum(1 for r in turns if r["status"] == k and inw(r["send"]))
    m["no_room_turns"] = len({(r["stream"], r["turn"]) for r in turns if r["status"] == "no_room"})
    answered = {(r["stream"], r["turn"]) for r in ok}
    m["refused_unsent"] = len({(r["stream"], r["turn"]) for r in turns if r["status"] in REFUSED} - answered)
    m["slot_waits"] = sum(1 for r in turns if slots and r["inflight"] >= slots) if slots else None
    m["slot_waits_window"] = sum(1 for r in turns if slots and r["inflight"] >= slots and inw(r["send"])) if slots else None
    sw = [r for r in turns if not r["cold"] and r["first"] is not None and inw(r["send"]) and r["first"] - r["send"] > slow]
    m["slow_warm"] = len(sw)
    m["slow_warm_full"] = sum(1 for r in sw if slots and r["inflight"] >= slots) if slots else None
    inf = []
    for u in range(nurl):
        ev = sorted([(max(r["send"], W), 1) for r in turns if r["u"] == u and r["end"] > W and r["send"] < E] +
                    [(min(r["end"], E), -1) for r in turns if r["u"] == u and r["end"] > W and r["send"] < E])
        area, cur, last, top = 0.0, 0, W, 0
        for t, dv in ev:
            area += cur * (t - last)
            cur += dv
            last = t
            top = max(top, cur)
        inf.append((area / M if M > 0 else None, top))
    m["inflight_url"] = inf
    firsts = {}
    for r in ok:
        if r["turn"] == 0:
            firsts[r["stream"]] = r["end"]
    m["warm_at_start"] = sum(1 for s in range(hdr["work"]["streams"]) if firsts.get(s, E + 1) < W)
    m["reply_med"] = med([r.get("completion_tokens") for r in ok])
    fin = {}
    for r in ok:
        fin[r.get("finish") or "?"] = fin.get(r.get("finish") or "?", 0) + 1
    m["finish"] = fin
    return m


def fmt(x, d=0):
    if x is None:
        return "-"
    return f"{x:,.{d}f}"


def block(path, hdr, m):
    w = hdr["work"]
    per = hdr.get("streams_per_url") or []
    out = [f"{path}: {hdr.get('label') or '-'}, started {hdr.get('started_utc')}, plan {hdr['plan_sha256'][:12]}, {w['streams']} streams on "
           f"{len(hdr['urls'])} URL{'s' if len(hdr['urls']) > 1 else ''} ({'/'.join(map(str, per))}), contexts {','.join(map(str, w['ctx']))} "
           f"up to {w['max_context']}, cold share {w['cold_share']}, max_tokens {w['max_tokens']}, thinking {'on' if w['thinking'] else 'off'}; "
           f"measured {fmt(m['window_s'], 0)} s from {fmt(m['window_start'], 0)} s (warm-up at least {fmt(hdr['warmup'], 0)} s, "
           f"until every first turn ended, at most {fmt(hdr.get('warmup_max') or hdr['warmup'], 0)} s)"]
    gu = " + ".join(fmt(x, 1) for x in m["gen_tps_url"]) if len(m["gen_tps_url"]) > 1 else ""
    out.append(f"  generated          {fmt(m['gen_tps'], 1)} tok/s" + (f" ({gu} by URL)" if gu else ""))
    out.append(f"  new prompt read    {fmt(m['new_prompt_tps'], 0)} tok/s; cached {fmt(m['cached_tokens'])} of {fmt(m['prompt_tokens'])} prompt tokens "
               f"({fmt(m['cached_tps'], 0)} tok/s); warm turns {fmt(None if m['warm_cached_share'] is None else 100 * m['warm_cached_share'], 1)}% cached, "
               f"{m['warm_resumed']} of {m['warm_reuse_n']} resumed the previous turn (cached >= 90% of its prompt and reply)")
    out.append(f"  new prompt : generated  {fmt(m['ratio'], 2)} : 1")
    out.append(f"  decode a stream    median {fmt(m['decode_cold_med'], 1)} tok/s while a cold read is in flight on its URL "
               f"({fmt(None if m['cold_time_share'] is None else 100 * m['cold_time_share'], 0)}% of decode time), {fmt(m['decode_none_med'], 1)} tok/s with none "
               f"(all streams pooled {fmt(m['decode_cold_all'], 1)} / {fmt(m['decode_none_all'], 1)})")
    out.append(f"  first token        warm p50/p90/p99 {fmt(m['ttft_warm_p50'], 2)}/{fmt(m['ttft_warm_p90'], 2)}/{fmt(m['ttft_warm_p99'], 2)} s "
               f"({m['ttft_warm_n']}); cold {fmt(m['ttft_cold_p50'], 2)}/{fmt(m['ttft_cold_p90'], 2)}/{fmt(m['ttft_cold_p99'], 2)} s ({m['ttft_cold_n']})")
    out.append(f"  token gaps         p99 {fmt(m['gap_p99'], 3)} s, max {fmt(m['gap_max'], 2)} s; {m['gaps_over_1s']} over 1 s")
    out.append(f"  turns              {m['turns_done']} done in the window ({m['cold_done']} cold, {m['forced_done']} of them because the window "
               f"was full); {m['turns_done_run']} in the run; planned sequence reached turn {m['reach_min']}/{fmt(m['reach_med'], 0)}/{m['reach_max']} "
               f"a stream (min/median/max), {m['reached_turns']} in all; reply median {fmt(m['reply_med'])} tokens, finish {m['finish']}")
    out.append(f"  refused, failed    no KV room {m['no_room']} attempts on {m['no_room_turns']} turns; busy {m['busy']}; too long {m['too_long']}; "
               f"errors {m['error']}; cut at the end {m['aborted']}; refused turns never answered {m['refused_unsent']}"
               + ("" if not m["no_usage"] else f"; {m['no_usage']} answers without usage"))
    sl = hdr.get("slots") or 0
    inf = "; ".join(f"URL {i}: mean {fmt(a, 1)} in flight, max {b}" for i, (a, b) in enumerate(m["inflight_url"]))
    out.append(f"  slots              " + (f"{sl} a URL; {m['slot_waits']} requests sent while all were busy (slot waits, {m['slot_waits_window']} in the window); "
                                          if sl else "not given (--slots); ") + inf +
               f"; {m['slow_warm']} warm turns waited over {os.environ.get('AL_SLOW_FIRST') or '10'} s for a first token"
               + (f" ({m['slow_warm_full']} of them sent while all slots were busy)" if sl else ""))
    out.append(f"  warm-up            {m['warm_at_start']} of {w['streams']} streams finished their first turn before measuring began"
               + ("" if m["warm_at_start"] == w["streams"] else " (the window opened before every stream was past its first cold read: raise --warmup-max)"))
    return "\n".join(out)


KEYS = [("gen_tps", "generated tok/s", 1), ("new_prompt_tps", "new prompt tok/s", 0), ("cached_tps", "cached prompt tok/s", 0),
        ("ratio", "new prompt : generated", 2), ("decode_cold_med", "decode tok/s a stream, cold read in flight", 1),
        ("decode_none_med", "decode tok/s a stream, none in flight", 1), ("ttft_warm_p50", "first token warm p50 s", 2),
        ("ttft_warm_p90", "first token warm p90 s", 2), ("ttft_warm_p99", "first token warm p99 s", 2),
        ("ttft_cold_p50", "first token cold p50 s", 2), ("ttft_cold_p90", "first token cold p90 s", 2),
        ("ttft_cold_p99", "first token cold p99 s", 2), ("gap_p99", "token gap p99 s", 3), ("gaps_over_1s", "token gaps over 1 s", 0),
        ("turns_done", "turns done in window", 0), ("reached_turns", "planned turns reached", 0), ("no_room", "no KV room refusals", 0),
        ("busy", "busy refusals", 0), ("too_long", "too long", 0), ("error", "errors", 0), ("slot_waits", "slot waits", 0),
        ("warm_at_start", "streams warm at start", 0)]

T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110,
        2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def tq(df):
    df = int(math.floor(df))
    return T975[df - 1] if 1 <= df <= 30 else 1.96


def meanci(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 3:
        return None
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    h = tq(n - 1) * sd / math.sqrt(n)
    return mu - h, mu + h


def welch(a, b):
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if len(a) < 3 or len(b) < 3:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1) / len(a)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return mb - ma, mb - ma
    df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1) if va else 0) + (vb ** 2 / (len(b) - 1) if vb else 0) or 1e-12)
    h = tq(df) * se
    return mb - ma - h, mb - ma + h


def cell(xs, d):
    m = med(xs)
    ci = meanci(xs)
    s = fmt(m, d)
    if ci:
        s += f" [{fmt(ci[0], d)}, {fmt(ci[1], d)}]"
    return s + f" ({len([x for x in xs if x is not None])})"


def runs(paths):
    out = []
    for p in paths:
        h, t, e = load(p)
        out.append((p, h, metrics(h, t, e)))
    return out


def same_work(groups):
    keyf = lambda h: (h["plan_sha256"], json.dumps(h["work"], sort_keys=True), h["warmup"], h.get("warmup_max"), h["measure"], h.get("turns"))
    ref = keyf(groups[0][0][1])
    bad = [p for g in groups for p, h, _ in g if keyf(h) != ref]
    return bad


def interleaving(groups, names):
    seq = sorted((h.get("started_utc") or "", n) for g, n in zip(groups, names) for _, h, _ in g)
    letters = [n for _, n in seq]
    longest, cur = 0, 0
    for i, x in enumerate(letters):
        cur = cur + 1 if i and letters[i - 1] == x else 1
        longest = max(longest, cur)
    return letters, longest


def report(paths):
    rs = runs(paths)
    for p, h, m in rs:
        print(block(p, h, m))
        print()
    groups = {}
    for r in rs:
        groups.setdefault((r[1]["plan_sha256"], r[1]["work"]["streams"], tuple(r[1]["urls"]), r[1].get("slots")), []).append(r)
    for (sha, s, urls, sl), g in groups.items():
        if len(g) < 2:
            continue
        print(f"across {len(g)} runs of plan {sha[:12]} ({s} streams) on {' '.join(urls)}: median [95% interval of the mean when 3+ runs] (runs)")
        for k, name, d in KEYS:
            print(f"  {name:44s} {cell([m.get(k) for _, _, m in g], d)}")
        print()
    return 0


def compare(a, b):
    ga, gb = runs(a), runs(b)
    bad = same_work([ga, gb])
    print(f"A: {len(ga)} runs ({', '.join(h.get('label') or p for p, h, _ in ga)}); B: {len(gb)} runs ({', '.join(h.get('label') or p for p, h, _ in gb)})")
    if bad:
        print(f"not the same work: {', '.join(bad)} differ from {ga[0][0]} in plan, settings or duration")
        if os.environ.get("AL_FORCE") != "1":
            return 1
    else:
        print(f"same work: plan {ga[0][1]['plan_sha256'][:12]}, same settings and duration on both sides")
    letters, longest = interleaving([ga, gb], ["A", "B"])
    print(f"order by start: {' | '.join(letters)}" + ("" if longest <= 2 else f" (not interleaved: {longest} runs of one side in a row)"))
    if min(len(ga), len(gb)) < 3:
        print("fewer than 3 runs a side: no intervals")
    print(f"  {'':44s} {'A median':>14s} {'B median':>14s} {'B/A':>7s}  95% interval of B-A (Welch)")
    for k, name, d in KEYS:
        xa, xb = [m.get(k) for _, _, m in ga], [m.get(k) for _, _, m in gb]
        ma, mb = med(xa), med(xb)
        ci = welch(xa, xb)
        ratio = f"{mb / ma:.3f}" if ma and mb is not None else "-"
        print(f"  {name:44s} {fmt(ma, d):>14s} {fmt(mb, d):>14s} {ratio:>7s}  " + (f"[{fmt(ci[0], d)}, {fmt(ci[1], d)}]" if ci else "-"))
    ta, tb = med([m["reached_turns"] for _, _, m in ga]), med([m["reached_turns"] for _, _, m in gb])
    if ta is not None and tb is not None and ta != tb:
        lo = "A" if ta < tb else "B"
        print(f"fell behind: {lo} reached {fmt(min(ta, tb))} of the planned turns against {fmt(max(ta, tb))} (median), so it did less of the same sequence")
    return 0


def table(arms):
    gs = [runs(fs) for _, fs in arms]
    names = [n for n, _ in arms]
    bad = same_work(gs)
    if bad:
        print(f"not the same work: {', '.join(bad)} differ from {gs[0][0][0]} in plan, settings or duration; such runs go in a separate table")
        if os.environ.get("AL_FORCE") != "1":
            return 1
    h = gs[0][0][1]
    print(f"plan {h['plan_sha256'][:12]}, {h['work']['streams']} streams, contexts {','.join(map(str, h['work']['ctx']))} up to {h['work']['max_context']}, "
          f"{fmt(h['measure'])} s measured; median [95% interval of the mean when 3+ runs] (runs)")
    letters, longest = interleaving(gs, names)
    print(f"order by start: {' | '.join(letters)}" + ("" if longest <= 2 else f" (not interleaved: {longest} runs of one arm in a row)"))
    print("| metric | " + " | ".join(f"{n} ({'/'.join(map(str, g[0][1].get('streams_per_url') or []))} streams a URL)" for n, g in zip(names, gs)) + " |")
    print("|---|" + "---:|" * len(gs))
    for k, name, d in KEYS:
        print(f"| {name} | " + " | ".join(cell([m.get(k) for _, _, m in g], d) for g in gs) + " |")
    return 0


def plan_cmd(o):
    pl = Plan(o)
    horizon = int(os.environ.get("AL_HORIZON") or "64")
    sha = pl.sha(horizon)
    nurl = max(1, len(o["url"]))
    print(f"plan {sha} (seed {o['seed']}, salt {o['salt']!r}, {o['streams']} streams, first {horizon} turns a stream, {TASKS_VERSION})")
    allt = [pl.turn(s, t) for s in range(o["streams"]) for t in range(horizon)]
    cold = [d for d in allt if d["cold"] and d["t"] > 0]
    print(f"turns after the first: {len(allt) - o['streams']}, cold {len(cold)} ({fmt(100.0 * len(cold) / max(1, len(allt) - o['streams']), 1)}%), "
          f"{sum(1 for d in cold if d.get('forced'))} of them because the conversation was full")
    dem = [d["est"] + o["max_tokens"] for d in allt]
    print(f"prompt plus reply budget a turn: median {fmt(med(dem))}, p90 {fmt(pct(dem, 90))}, max {fmt(max(dem))} tokens (limit {o['max_context']})")
    firsts = [sum(pl.turn(s, 0)["est"] for s in range(o["streams"]) if s % nurl == u) for u in range(nurl)]
    print(f"first turns, read cold before measuring: {' + '.join(fmt(x) for x in firsts)} tokens by URL; at 3,000 tok/s a URL the "
          f"slowest URL takes {fmt(max(firsts) / 3000.0)} s")
    for s in range(min(o["streams"], 4)):
        print(f"  stream {s}: " + " ".join(("C" if d["cold"] else "w") + str(round(d["est"] / 1000)) + "k" for d in (pl.turn(s, t) for t in range(min(horizon, 16)))))
    if o["kv_tokens"]:
        r = G.R(o["seed"], "agentload-fit")
        for u in range(nurl):
            ss = [s for s in range(o["streams"]) if s % nurl == u]
            fit, tot = 0, 4000
            for _ in range(tot):
                fit += sum(pl.turn(s, r.below(horizon))["est"] + o["max_tokens"] for s in ss) <= o["kv_tokens"]
            mean = sum(med([pl.turn(s, t)["est"] + o["max_tokens"] for t in range(horizon)]) for s in ss)
            print(f"URL {u}: {len(ss)} streams, KV pool {fmt(o['kv_tokens'])} tokens ({fmt(o['kv_tokens'] / max(1, len(ss)))} a stream): "
                  f"all {len(ss)} fit at once in {fmt(100.0 * fit / tot, 1)}% of random draws of one planned turn a stream; "
                  f"sum of the streams' median budgets {fmt(mean)}; worst case {fmt(len(ss) * o['max_context'])}")
    return 0


def verify_cmd(o):
    rc = 0
    m = G.load(os.environ.get("LCQ_MANIFEST") or G.MANIFEST)
    rc |= G.verify(os.environ.get("LCQ_MANIFEST") or G.MANIFEST)
    c = G.corpus()
    errs = []
    for it in m["items"]:
        pre, _ = G.build(it)
        order = G.order(it)
        bl = [c[order[(it["start"] + i) % len(c)]] for i in range(it["nblocks"])]
        e = sum(b["n"] for b in bl) + est_text("x" * (len(pre) - sum(len(b["t"]) for b in bl)))
        errs.append((e - it["tokens"]) / it["tokens"])
    ok = min(errs) > -0.005
    rc |= 0 if ok else 1
    print(f"GATE\tagentload-estimate\t{'PASS' if ok else 'FAIL'} (token estimate against the {len(errs)} planned haystacks' tokenizer counts: "
          f"{100 * min(errs):+.2f}% to {100 * max(errs):+.2f}%; a high estimate only leaves more room)")
    o2 = dict(o)
    a, b = Plan(o).sha(16), Plan(o2).sha(16)
    same = a == b
    pl = Plan(o)
    solved, tried = 0, 0
    for s in range(min(o["streams"], 4)):
        d = pl.turn(s, 0)
        pre, tasks = G.build(d["item"])
        for t in tasks[:4]:
            tried += 1
            ans, hits = G.solve(pre + G.question(t))
            solved += G.score(t, ans)[1] is True and hits == 1
    good = same and solved == tried
    rc |= 0 if good else 1
    print(f"GATE\tagentload-plan\t{'PASS' if good else 'FAIL'} (plan built twice: {'same' if same else 'different'} sha256 {a[:12]}; "
          f"{solved} of {tried} needles in the first contexts have exactly one reading)")
    return 1 if rc else 0


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    try:
        if cmd == "report" and len(argv) > 2:
            return report(argv[2:])
        if cmd == "compare" and "--" in argv:
            i = argv.index("--")
            if i > 2 and i < len(argv) - 1:
                return compare(argv[2:i], argv[i + 1:])
        if cmd == "table" and len(argv) > 3 and argv[2] == "--arm":
            arms = []
            for a in argv[2:]:
                if a == "--arm":
                    arms.append([None, []])
                elif arms[-1][0] is None:
                    arms[-1][0] = a
                else:
                    arms[-1][1].append(a)
            if all(n and fs for n, fs in arms):
                return table(arms)
        if cmd in ("run", "plan", "show", "verify"):
            pos, o = parse(argv[2:])
            if cmd == "plan" and not pos:
                return plan_cmd(o)
            if cmd == "verify" and not pos:
                return verify_cmd(o)
            if cmd == "show" and len(pos) == 2:
                pl = Plan(o)
                msgs = planned_messages(pl, int(pos[0]), int(pos[1]))
                sys.stdout.write(body(dict(o, model=o["model"] or "MODEL"), msgs).decode() + "\n")
                return 0
            if cmd == "run" and len(pos) == 1:
                if not o["url"] or not o["model"]:
                    raise ArgError("run needs --url and --model")
                return Run(o, pos[0]).run()
    except ArgError as e:
        if str(e):
            print(f"agentload: {e}", file=sys.stderr)
            return 2
    except (OSError, ValueError) as e:
        print(f"agentload: {e}", file=sys.stderr)
        return 2
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
