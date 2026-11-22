#!/usr/bin/env python3
import gzip
import hashlib
import json
import os
import re
import sys

USAGE = """lcqgen.py: LCQ v2, a seeded long-context task set: several independent haystacks a depth, each with needles.

A haystack is a unique session header, then filler framed as agent traffic (read_file tool results and assistant notes
holding public-domain book text: corpus/corpus.jsonl.gz, rebuilt by fetch_corpus.py) with needles placed inside it, up
to a target depth in tokens. Every haystack carries, at fixed relative positions:
  6 record lookups: a service registry record (JSON tool result, YAML note or prose note) among 3 near-duplicate
    records each (the id one digit off or two digits swapped, other fields different); asked for two fields;
  2 joins: an employee record names a project, the project's record names a cost center; the two records sit far
    apart (0.12 and 0.70, 0.88 and 0.33), with 2 near-duplicate employees and 2 near-duplicate projects each;
  4 passages of 6 generated sentences between BEGIN/END lines, each with a near-duplicate passage (same first
    sentence, other sentences new, id one character off); each asked twice: copy word for word, and copy in capitals.
A question follows the haystack, so every question on one haystack shares its whole prefix.

The plan has HAYS haystacks a depth (4 at 131,072-400,000, 2 at 8,192-95,000: 64 and 32 tasks a depth), seed 8181.
Haystack i of a depth is item h<depth>-<i>, idx i: its filler start, every needle id, every near-duplicate and every
distractor position, and the order of the corpus blocks it draws filler from (a seeded Fisher-Yates permutation of all
blocks), come from splitmix64 keyed by (seed, kind, depth, i). The corpus holds about 1.0M tokens, so four 400K
haystacks reuse blocks, but each in its own order: the text around a needle differs between haystacks. Ids are unique
within a haystack (the solver proves one exact-id reading a task). The needles of a task kind sit at the same relative
depth in every haystack.

The plan (manifest.json) fixes each item's corpus start, block count and tail lines so its prefix hits the depth in
the model's tokens, counted offline with its tokenizer; it also holds each prefix's sha256 and token count, and the
sha256 of the uncompressed corpus. Only `plan` needs the tokenizer (the tokenizers package); everything else is the
standard library.

  lcqgen.py plan MANIFEST TOKENIZER.json   write the plan
  lcqgen.py verify [MANIFEST]              rebuild every item, compare sha256 and the solver: TSV lines
                                           lcq-gen (every prefix's sha256 as planned) and lcq-solve (every task's
                                           expected answer is the only one an exact-id reading of the text gives)
  lcqgen.py show MANIFEST ITEM [TASK]      print an item's prompt (prefix and the task's question)

Scoring (score()): lookup and join 0.5 a field, exact when both match; copy and caps 1 - word edit distance / words,
exact when the words are equal. Letter grade of a mean score: A >= 0.95, B >= 0.85, C >= 0.70, D >= 0.50, else F."""

HERE = os.path.dirname(os.path.realpath(__file__))
CORPUS = os.path.join(HERE, "corpus", "corpus.jsonl.gz")
MANIFEST = os.path.join(HERE, "manifest.json")
SEED = 8181
HAYS = {8192: 2, 32768: 2, 65536: 2, 95000: 2, 131072: 4, 200000: 4, 262144: 4, 330000: 4, 400000: 4}
DEPTHS = [8192, 32768, 65536, 95000, 131072, 200000, 262144, 330000, 400000]



QRES = 160
LOOK_AT = [0.04, 0.22, 0.41, 0.59, 0.78, 0.96]
JOIN_AT = [(0.12, 0.70), (0.88, 0.33)]
PASS_AT = [0.08, 0.30, 0.52, 0.84]
M64 = 0xFFFFFFFFFFFFFFFF

WORDS = """amber anchor apple arch arrow atlas autumn badge bamboo banner barrel basin beacon beetle bell birch blanket
blossom bottle boulder bracket breeze brick bridge bronze brook bucket button cabin cable cactus camera candle canyon
canvas carbon carpet castle cedar cellar chalk channel chapter cherry chimney circle cliff clock cloud clover cobalt
comet compass copper coral cotton crane crater crystal cupboard current curtain cushion dagger daisy delta desert
diamond dolphin donkey dragon drawer drum eagle echo ember engine falcon feather fence fern ferry fiddle field fig
flame flask fleet forest fossil fountain fox frost garden garnet gate glacier glove granite grape gravel harbor
harvest hazel helmet heron hollow honey horizon island ivory jacket jasmine jungle kettle kite ladder lagoon lantern
laurel lemon lighthouse lily linen locket magnet mango maple marble meadow mirror mitten monsoon mosaic moss mountain
needle nest nickel oak oasis ocean olive orchard otter oyster paddle palace panther parcel pebble pepper pigeon pillar
pine planet plaza pocket pond poppy prism pumpkin quarry quill rabbit radish raven reef ribbon ridge river rocket
saddle salmon satchel scarf seashell shadow shelf shovel signal silver slate sparrow spindle spruce stable star stone
storm summit swan tablet teapot thistle thunder timber tower trellis tulip tunnel valley velvet violet walnut wagon
whistle willow window winter wren yarrow zephyr""".split()
ADJ = """quiet bright narrow gentle hollow rapid silent golden crooked distant ancient steady patient brittle scarlet
humble restless tidy weary eager frozen woven rustic polished hidden sudden tender vivid dusty gleaming loyal""".split()
VERB = """carries follows guards paints measures gathers folds shelters lifts sorts traces mends polishes weighs
borrows counts warms tilts sketches stacks rinses trims""".split()
PREP = "near beside under above behind across beyond toward against inside".split()
FIRST = """Mara Ilse Tomas Rafe Juno Odile Kasper Nell Bruno Vera Sorin Lida Anouk Piet Greta Hugo Ines Matti Rosa
Emil""".split()
LAST = """Quist Halvorsen Okafor Brandt Moreau Szabo Lindqvist Duarte Petrov Achterberg Novak Ferreira Kowalczyk
Haugen Ricci Varga Delacroix Ostrom Mbeki Sato""".split()
REGIONS = """eu-north-1 eu-north-2 eu-west-1 eu-west-3 eu-central-1 us-east-1 us-east-2 us-west-1 us-west-2 ap-south-1
ap-east-2 ap-northeast-1 sa-east-1 ca-central-1 af-south-1""".split()
TEAMS = "heron lynx otter falcon badger raven marten ibis crane osprey".split()
SVC = "billing ledger relay gateway indexer mailer scheduler catalog search notifier auth export importer webhook".split()
SVC2 = "relay worker proxy cache sync api daemon bridge feed queue".split()
PROJ = """ledger cleanup, invoice backfill, search reindex, mobile onboarding, audit trail, cache warmup, quota rework,
schema split, alert routing, export rewrite""".replace("\n", " ").split(", ")


class R:
    def __init__(self, *key):
        self.s = int(hashlib.sha256(repr(key).encode()).hexdigest()[:16], 16)

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def pick(self, xs):
        return xs[self.below(len(xs))]

    def digits(self, n):
        return "".join(str(self.below(10)) for _ in range(n))


_corpus = None


def corpus_sha():
    with gzip.open(CORPUS, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def corpus():
    global _corpus
    if _corpus is None:
        with gzip.open(CORPUS, "rt", encoding="ascii") as f:
            _corpus = [json.loads(l) for l in f]
    return _corpus


def neardup(r, digits, taken):
    for _ in range(200):
        d = list(digits)
        if r.below(2):
            i = r.below(len(d))
            d[i] = str((int(d[i]) + 1 + r.below(9)) % 10)
        else:
            i = r.below(len(d) - 1)
            if d[i] == d[i + 1]:
                continue
            d[i], d[i + 1] = d[i + 1], d[i]
        x = "".join(d)
        if x != digits and x not in taken:
            taken.add(x)
            return x
    raise RuntimeError("no near-duplicate id")


def fresh(r, n, taken):
    while True:
        x = r.digits(n)
        if x[0] != "0" and x not in taken:
            taken.add(x)
            return x


def svc_rec(r, rec):
    f = r.below(3)
    if f == 0:
        return (f"tool: http GET /registry/services/{rec['id']} -> 200\n"
                + json.dumps({"id": rec["id"], "name": rec["name"], "owner": rec["owner"], "port": rec["port"], "region": rec["region"]}) + "\n\n")
    if f == 1:
        return (f"assistant: Registry entry I found while tracing the deploy:\n  id: {rec['id']}\n  name: {rec['name']}\n"
                f"  owner: {rec['owner']}\n  port: {rec['port']}\n  region: {rec['region']}\n\n")
    return (f"assistant: Note for later: service {rec['id']} ({rec['name']}) belongs to {rec['owner']}, listens on port "
            f"{rec['port']} and runs in {rec['region']}.\n\n")


def emp_rec(r, rec):
    f = r.below(3)
    if f == 0:
        return "tool: http GET /people/" + rec["id"] + " -> 200\n" + json.dumps({"employee": rec["id"], "name": rec["name"], "project": rec["project"]}) + "\n\n"
    if f == 1:
        return f"assistant: Staffing sheet row:\n  employee: {rec['id']}\n  name: {rec['name']}\n  project: {rec['project']}\n\n"
    return f"assistant: From the standup notes: {rec['id']} ({rec['name']}) is assigned to project {rec['project']} now.\n\n"


def prj_rec(r, rec):
    f = r.below(3)
    if f == 0:
        return "tool: http GET /projects/" + rec["id"] + " -> 200\n" + json.dumps({"project": rec["id"], "title": rec["title"], "cost_center": rec["cc"]}) + "\n\n"
    if f == 1:
        return f"assistant: Project card:\n  project: {rec['id']}\n  title: {rec['title']}\n  cost_center: {rec['cc']}\n\n"
    return f"assistant: Finance confirmed that project {rec['id']} ({rec['title']}) is billed to cost center {rec['cc']}.\n\n"


def sentence(r):
    return (f"The {r.pick(ADJ)} {r.pick(WORDS)} {r.pick(VERB)} the {r.pick(WORDS)} {r.pick(PREP)} the {r.pick(ADJ)} "
            f"{r.pick(WORDS)}, and every {r.pick(WORDS)} {r.pick(VERB)} a {r.pick(WORDS)} {r.pick(PREP)} the {r.pick(WORDS)}.")


def pid(r, taken):
    al = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        x = "".join(r.pick(al) for _ in range(3))
        if x not in taken:
            taken.add(x)
            return x


def pid_near(r, x, taken):
    al = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        i = r.below(3)
        y = x[:i] + r.pick(al) + x[i + 1:]
        if y != x and y not in taken:
            taken.add(y)
            return y


def passage_block(xid, sents):
    return f"assistant: Draft paragraph for the release notes:\nBEGIN PASSAGE PSG-{xid}\n{' '.join(sents)}\nEND PASSAGE PSG-{xid}\n\n"


def needles(seed, kind, depth, idx):
    r = R(seed, kind, depth, idx, "needles")
    taken, ptaken = set(), set()
    ins, tasks = [], []
    for k, at in enumerate(LOOK_AT):
        d = fresh(r, 5, taken)
        rec = {"id": "SVC-" + d, "name": f"{r.pick(SVC)}-{r.pick(SVC2)}", "owner": "team-" + r.pick(TEAMS),
               "port": 1024 + r.below(64000), "region": r.pick(REGIONS)}
        ins.append((at, svc_rec(r, rec)))
        for _ in range(3):
            dup = dict(rec, id="SVC-" + neardup(r, d, taken), port=1024 + r.below(64000), region=r.pick([x for x in REGIONS if x != rec["region"]]),
                       owner="team-" + r.pick(TEAMS))
            ins.append((r.below(10000) / 10000.0, svc_rec(r, dup)))
        tasks.append({"task": f"look{k}", "kind": "lookup", "pos": at, "id": rec["id"],
                      "expect": {"port": str(rec["port"]), "region": rec["region"]}, "max_tokens": 48})
    for k, (ea, pa) in enumerate(JOIN_AT):
        e, p = fresh(r, 5, taken), fresh(r, 3, taken)
        cc = "CC-" + fresh(r, 5, taken)
        emp = {"id": "E-" + e, "name": f"{r.pick(FIRST)} {r.pick(LAST)}", "project": "PRJ-" + p}
        prj = {"id": "PRJ-" + p, "title": r.pick(PROJ), "cc": cc}
        ins.append((ea, emp_rec(r, emp)))
        ins.append((pa, prj_rec(r, prj)))
        for _ in range(2):
            p2 = neardup(r, p, taken)
            ins.append((r.below(10000) / 10000.0, emp_rec(r, {"id": "E-" + neardup(r, e, taken), "name": f"{r.pick(FIRST)} {r.pick(LAST)}", "project": "PRJ-" + p2})))
            ins.append((r.below(10000) / 10000.0, prj_rec(r, {"id": "PRJ-" + p2, "title": r.pick(PROJ), "cc": "CC-" + fresh(r, 5, taken)})))
            ins.append((r.below(10000) / 10000.0, prj_rec(r, {"id": "PRJ-" + neardup(r, p, taken), "title": r.pick(PROJ), "cc": "CC-" + fresh(r, 5, taken)})))
        tasks.append({"task": f"join{k}", "kind": "join", "pos": min(ea, pa), "pos2": max(ea, pa), "id": emp["id"],
                      "expect": {"project": prj["id"], "cost_center": cc}, "max_tokens": 48})
    for k, at in enumerate(PASS_AT):
        x = pid(r, ptaken)
        sents = [sentence(r) for _ in range(6)]
        ins.append((at, passage_block(x, sents)))
        ins.append((r.below(10000) / 10000.0, passage_block(pid_near(r, x, ptaken), [sents[0]] + [sentence(r) for _ in range(5)])))
        text = " ".join(sents)
        tasks.append({"task": f"copy{k}", "kind": "copy", "pos": at, "id": "PSG-" + x, "expect": {"text": text}, "max_tokens": 320})
        tasks.append({"task": f"caps{k}", "kind": "caps", "pos": at, "id": "PSG-" + x, "expect": {"text": text.upper()}, "max_tokens": 320})
    return ins, tasks


def header(seed, kind, depth, idx):
    h = hashlib.sha256(repr((seed, kind, depth, idx, "head")).encode()).hexdigest()[:12]
    return f"session {h} (replay of an agent's work session: tool results and notes, {kind} {depth} {idx})\n\n"


def tail_text(block, lines):
    if lines <= 0:
        return ""
    t = block["t"].rstrip("\n").split("\n")[:lines]
    t = "\n".join(t)
    if t.count("```") % 2:
        t += "\n```"
    return t + "\n\n"


_orders = {}


def order(item):
    key = (item["seed"], item["kind"], item["depth"], item["idx"])
    if key not in _orders:
        n = len(corpus())
        o = list(range(n))
        r = R(*key, "order")
        for i in range(n - 1, 0, -1):
            j = r.below(i + 1)
            o[i], o[j] = o[j], o[i]
        _orders[key] = o
    return _orders[key]


def build(item):
    c = corpus()
    n = len(c)
    seed, kind, depth, idx = item["seed"], item["kind"], item["depth"], item["idx"]
    o = order(item)
    blocks = [c[o[(item["start"] + i) % n]]["t"] for i in range(item["nblocks"])]
    tail = tail_text(c[o[(item["start"] + item["nblocks"]) % n]], item["tail"])
    tasks = []
    if kind != "pf":
        ins, tasks = needles(seed, kind, depth, idx)
        slots = [[] for _ in range(len(blocks) + 1)]
        for at, t in ins:
            slots[min(len(blocks), int(round(at * len(blocks))))].append(t)
        out = [header(seed, kind, depth, idx)]
        for i, b in enumerate(blocks):
            out.extend(slots[i])
            out.append(b)
        out.append(tail)
        out.extend(slots[len(blocks)])
        prefix = "".join(out)
    else:
        prefix = header(seed, kind, depth, idx) + "".join(blocks) + tail
    return prefix, tasks


def question(t):
    k = t.get("kind")
    if k == "lookup":
        return (f"Question: The material above is a work session transcript. Somewhere in it is the service registry record for "
                f"service {t['id']}; there are similar records for other services, so use only the one whose id is exactly "
                f"{t['id']}. Reply with one line in exactly this form and nothing else:\nport=<port> region=<region>")
    if k == "join":
        return (f"Question: In the material above, employee {t['id']} is assigned to exactly one project, and that project's own "
                f"record gives its cost center. Use only records whose ids match exactly. Reply with one line in exactly this "
                f"form and nothing else:\nproject=<project id> cost_center=<cost center>")
    if k == "copy":
        return (f"Question: Passage {t['id']} appears in the material above between the lines 'BEGIN PASSAGE {t['id']}' and "
                f"'END PASSAGE {t['id']}' (a similar passage with another id also appears; use only {t['id']}). Reproduce its "
                f"text exactly, word for word, with nothing before or after it.")
    if k == "caps":
        return (f"Question: Passage {t['id']} appears in the material above between the lines 'BEGIN PASSAGE {t['id']}' and "
                f"'END PASSAGE {t['id']}' (a similar passage with another id also appears; use only {t['id']}). Reproduce its "
                f"text with every letter written as a capital letter, keeping every word, its order and the punctuation, with "
                f"nothing before or after it.")
    return "Question: Reply with the single word OK."


def strip_think(text):
    text = text or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def words(s):
    return s.split()


def wdist(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def clean_copy(text, pid_):
    lines = [l for l in text.replace("```", "\n").split("\n") if pid_ not in l]
    s = " ".join(lines).strip()
    if len(s) > 1 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1]
    return s


def score(t, text):
    text = strip_think(text)
    k = t["kind"]
    if k == "lookup":
        p = re.search(r"port\s*[=:]\s*(\d+)", text)
        g = re.search(r"region\s*[=:]\s*([a-z]{2}-[a-z]+-\d)", text)
        got = {"port": p.group(1) if p else None, "region": g.group(1) if g else None}
    elif k == "join":
        p = re.search(r"project\s*[=:]\s*(PRJ-\d+)", text)
        g = re.search(r"cost_center\s*[=:]\s*(CC-\d+)", text)
        got = {"project": p.group(1) if p else None, "cost_center": g.group(1) if g else None}
    elif k in ("copy", "caps"):
        ref = words(t["expect"]["text"])
        gw = words(clean_copy(text, t["id"]))
        d = wdist(ref, gw)
        sc = max(0.0, 1.0 - d / max(1, len(ref)))
        return round(sc, 4), gw == ref, {"edits": d, "words": len(ref), "got_words": len(gw)}
    else:
        return 1.0, True, {}
    hits = sum(1 for f, v in t["expect"].items() if got.get(f) == v)
    return round(hits / len(t["expect"]), 4), hits == len(t["expect"]), {"got": got}


def grade(m):
    if m is None:
        return "-"
    return "A" if m >= 0.95 else "B" if m >= 0.85 else "C" if m >= 0.70 else "D" if m >= 0.50 else "F"


def solve(text):
    q = text[text.rfind("Question: "):]
    body = text[:text.rfind("Question: ")]
    m = re.search(r"service (SVC-\d+)", q)
    if m:
        i = re.escape(m.group(1))
        for pat in (r'"id": "%s", "name": "[^"]*", "owner": "[^"]*", "port": (\d+), "region": "([^"]+)"' % i,
                    r"id: %s\n  name: [^\n]*\n  owner: [^\n]*\n  port: (\d+)\n  region: (\S+)" % i,
                    r"service %s \([^)]*\) belongs to [^,]*, listens on port (\d+) and runs in ([a-z0-9-]+)\." % i):
            hits = re.findall(pat, body)
            if hits:
                return f"port={hits[0][0]} region={hits[0][1]}", len(hits)
        return "", 0
    m = re.search(r"employee (E-\d+)", q)
    if m:
        i = re.escape(m.group(1))
        pr = re.findall(r'"employee": "%s", "name": "[^"]*", "project": "(PRJ-\d+)"|employee: %s\n  name: [^\n]*\n  project: (PRJ-\d+)|%s \([^)]*\) is assigned to project (PRJ-\d+) ' % (i, i, i), body)
        ps = sorted({x for h in pr for x in h if x})
        if len(ps) != 1:
            return "", len(ps)
        j = re.escape(ps[0])
        cc = re.findall(r'"project": "%s", "title": "[^"]*", "cost_center": "(CC-\d+)"|project: %s\n  title: [^\n]*\n  cost_center: (CC-\d+)|project %s \([^)]*\) is billed to cost center (CC-\d+)\.' % (j, j, j), body)
        cs = sorted({x for h in cc for x in h if x})
        if len(cs) != 1:
            return "", len(cs)
        return f"project={ps[0]} cost_center={cs[0]}", len(pr) * len(cc)
    m = re.search(r"Passage (PSG-[0-9A-Z]{3}) appears", q)
    if m:
        i = re.escape(m.group(1))
        hits = re.findall(r"BEGIN PASSAGE %s\n([^\n]*)\nEND PASSAGE %s\n" % (i, i), body)
        if not hits:
            return "", 0
        return (hits[0].upper() if "capital letter" in q else hits[0]), len(hits)
    return "OK", 1


def plan_items():
    return [{"id": f"h{d}-{i}", "seed": SEED, "kind": "hay", "depth": d, "idx": i} for d in DEPTHS for i in range(HAYS[d])]


def plan(path, tokp):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokp)
    count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    c = corpus()
    n = len(c)
    out = []
    for it in plan_items():
        target = it["depth"] - (QRES if it["kind"] != "pf" else 24)
        it = dict(it, start=R(it["seed"], it["kind"], it["depth"], it["idx"], "start").below(n), nblocks=0, tail=0)
        acc = count(build(it)[0])
        nb = 0
        o = order(it)
        while acc + c[o[(it["start"] + nb) % n]]["n"] <= target:
            acc += c[o[(it["start"] + nb) % n]]["n"]
            nb += 1
        it["nblocks"] = nb
        lo, hi = 0, len(c[o[(it["start"] + nb) % n]]["t"].split("\n"))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            it["tail"] = mid
            if count(build(it)[0]) <= target:
                lo = mid
            else:
                hi = mid - 1
        it["tail"] = lo
        prefix, tasks = build(it)
        it["tokens"] = count(prefix)
        it["sha256"] = hashlib.sha256(prefix.encode()).hexdigest()
        it["q_tokens"] = {t["task"]: count(prefix + question(t)) - it["tokens"] for t in tasks} if tasks else {"ok": count(prefix + question({"kind": "pf"})) - it["tokens"]}
        out.append(it)
        print(f"{it['id']}: start {it['start']} blocks {nb} tail {it['tail']} tokens {it['tokens']} (target {target})", flush=True)
    corp = corpus_sha()
    tsha = hashlib.sha256(open(tokp, "rb").read()).hexdigest()
    with open(path, "w") as f:
        json.dump({"seed": SEED, "corpus_sha256": corp, "corpus_blocks": n, "corpus_tokens": sum(b["n"] for b in c),
                   "tokenizer": os.path.basename(tokp), "tokenizer_sha256": tsha, "qres": QRES, "items": out}, f, indent=1)
    return 0


def load(path):
    with open(path) as f:
        return json.load(f)


def verify(path):
    m = load(path)
    corp = corpus_sha()
    bad, unsolved, n_t = [], [], 0
    if corp != m["corpus_sha256"]:
        bad.append(f"corpus {corp[:12]} not {m['corpus_sha256'][:12]}")
    else:
        for it in m["items"]:
            prefix, tasks = build(it)
            h = hashlib.sha256(prefix.encode()).hexdigest()
            if h != it["sha256"]:
                bad.append(f"{it['id']} {h[:12]} not {it['sha256'][:12]}")
            for t in tasks:
                n_t += 1
                ans, hits = solve(prefix + question(t))
                if score(t, ans)[1] is not True or hits != 1:
                    unsolved.append(f"{it['id']}/{t['task']} ({hits} readings)")
    items = len(m["items"])
    print("GATE\tlcq-gen\t" + (f"FAIL ({len(bad)} of {items} items differ from the plan: {'; '.join(bad[:4])})" if bad else
                               f"PASS ({items} items rebuilt here match the plan's sha256: seed {m['seed']}, corpus {corp[:12]}, {sum(i['tokens'] for i in m['items'])} prefix tokens in all)"))
    if not bad:
        print("GATE\tlcq-solve\t" + (f"FAIL ({len(unsolved)} of {n_t} tasks have no single exact-id reading: {'; '.join(unsolved[:4])})" if unsolved else
                                     f"PASS (each of {n_t} tasks has exactly one exact-id reading in its text, and it is the expected answer)"))
    return 1 if bad or unsolved else 0


def main(argv):
    if len(argv) < 2 or argv[1] not in ("plan", "verify", "show"):
        print(USAGE, file=sys.stderr)
        return 2
    if argv[1] == "plan" and len(argv) == 4:
        return plan(argv[2], argv[3])
    if argv[1] == "verify" and len(argv) in (2, 3):
        return verify(argv[2] if len(argv) == 3 else MANIFEST)
    if argv[1] == "show" and len(argv) in (4, 5):
        it = next((i for i in load(argv[2])["items"] if i["id"] == argv[3]), None)
        if it is None:
            print(f"lcqgen: no item {argv[3]}", file=sys.stderr)
            return 2
        prefix, tasks = build(it)
        t = next((x for x in tasks if x["task"] == argv[4]), None) if len(argv) == 5 else None
        sys.stdout.write(prefix + (question(t) if t else question({"kind": "pf"})) + "\n")
        if t:
            print(json.dumps(t), file=sys.stderr)
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
