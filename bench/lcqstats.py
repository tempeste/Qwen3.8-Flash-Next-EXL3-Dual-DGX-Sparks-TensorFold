#!/usr/bin/env python3
import json
import math
import os
import sys

USAGE = """lcqstats.py: LCQ v2 scores with their uncertainty, and which differences between arms are real.

  lcqstats.py table OUT ARM [ARM ...]   the report (stdout) and TSV gate lines (fd 3 when open, else stderr)

Reads OUT/<ARM>.lcq.jsonl (or OUT/perf-<ARM>.lcq.jsonl), each from lcq.py run. A task is (item, task): the same 16 questions on the same planned
haystacks for every arm, so arms pair task by task.

Per arm and depth, over the temperature-0 tasks (phases cold and warm):
  n, exact k/n with its Wilson 95% interval, mean score, and each haystack's exact count (spread across haystacks);
  haystacks lcq.py left 'unreached' are named with the reason.
Paired, per pair of arms and depth, and pooled over 131,072-400,000:
  exact: McNemar's exact test on the discordant tasks (b: first arm exact and second not; c: the reverse), two-sided
  binomial p of min(b, c) in b + c at 1/2; score: exact sign test on the per-task score differences (ties dropped);
  p values Holm-adjusted over every test of that pair; a difference is REAL when its adjusted p < 0.05, else NOT SHOWN
  (the data do not tell the arms apart at this size), with the discordant counts so the size of what was missed shows.
Sampled spread (phase samp, temperature > 0, reps 1..R on a subset): each rep's exact k/n with Wilson 95%, the share
of tasks whose exact outcome agrees between reps, McNemar rep 1 against rep 2, and the temperature-0 result on the
same tasks, so a temperature-0 difference between arms can be set against the run-to-run spread of one arm.
Exit 0; 2 on bad arguments or no records at all."""

POOL = (131072, 200000, 262144, 330000, 400000)
Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [1.0] * len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def load(out, arm):
    p = os.path.join(out, f"{arm}.lcq.jsonl")
    if not os.path.exists(p):
        p = os.path.join(out, f"perf-{arm}.lcq.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def t0(rows):
    return {(r["item"], r["task"]): r for r in rows if r.get("phase") in ("cold", "warm")}


NAMES = {8192: "8K", 32768: "32K", 65536: "65K", 95000: "95K", 131072: "131K", 200000: "200K", 262144: "262K", 330000: "330K", 400000: "400K"}


def kname(d):
    return NAMES.get(d, f"{round(d / 1000)}K")


def fmt_ci(k, n):
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def exact_of(rs):
    return sum(1 for r in rs if r.get("exact") is True)


def arm_table(arm, rows, emit):
    lines = [f"{arm}: temperature 0, exact k/n [Wilson 95%], mean score, exact a haystack"]
    by = t0(rows)
    unr = [r for r in rows if r.get("phase") == "unreached"]
    depths = sorted({r["depth"] for r in by.values()})
    for d in depths:
        rs = [r for r in by.values() if r["depth"] == d]
        ms = sum(float(r.get("score") or 0) for r in rs) / len(rs)
        hays = {}
        for r in rs:
            h = hays.setdefault(r.get("hay", 0), [0, 0])
            h[1] += 1
            h[0] += r.get("exact") is True
        hs = " ".join(f"{a}/{b}" for _, (a, b) in sorted(hays.items()))
        errs = sum(1 for r in rs if r.get("error"))
        lines.append(f"  {kname(d):>5s}  {fmt_ci(exact_of(rs), len(rs)):32s} score {ms:.3f}  hays {hs}" + (f"  ({errs} errors scored 0)" if errs else ""))
    for r in unr:
        lines.append(f"  {kname(r['depth']):>5s}  {r['item']} unreached: {r.get('why', '')}")
    pn = [r for r in by.values() if r["depth"] in POOL]
    if pn:
        lines.append(f"  pool 131K-400K  {fmt_ci(exact_of(pn), len(pn))}")
    if not depths:
        emit(f"lcqstat-{arm}", "FAIL (no temperature-0 records)")
    else:
        msg = "; ".join(f"{kname(d)} " + fmt_ci(exact_of([r for r in by.values() if r['depth'] == d]), sum(1 for r in by.values() if r['depth'] == d)) for d in depths)
        if unr:
            ds = sorted({r["depth"] for r in unr})
            msg += f"; unreached haystacks at {', '.join(kname(d) for d in ds)} ({unr[0].get('why', '')})"
        emit(f"lcqstat-{arm}", f"INFO ({msg})")
    return lines


def sign_test(diffs):
    pos = sum(1 for x in diffs if x > 1e-9)
    neg = sum(1 for x in diffs if x < -1e-9)
    return pos, neg, binom_two_sided(pos, neg)


def pair_table(a, ra, b, rb, emit):
    lines = [f"{a} against {b}: paired by task (same haystack, same question); b = {a} exact and {b} not, c = the reverse"]
    ta, tb = t0(ra), t0(rb)
    keys = sorted(set(ta) & set(tb), key=lambda k: (ta[k]["depth"], k))
    depths = sorted({ta[k]["depth"] for k in keys})
    tests = []
    for d in depths + ["pool"]:
        ks = [k for k in keys if (ta[k]["depth"] in POOL if d == "pool" else ta[k]["depth"] == d)]
        if not ks:
            continue
        bb = sum(1 for k in ks if ta[k].get("exact") is True and tb[k].get("exact") is not True)
        cc = sum(1 for k in ks if ta[k].get("exact") is not True and tb[k].get("exact") is True)
        pos, neg, ps = sign_test([float(ta[k].get("score") or 0) - float(tb[k].get("score") or 0) for k in ks])
        tests.append({"d": d, "n": len(ks), "b": bb, "c": cc, "ea": exact_of([ta[k] for k in ks]), "eb": exact_of([tb[k] for k in ks]),
                      "p": binom_two_sided(bb, cc), "pos": pos, "neg": neg, "ps": ps})
    adj = holm([t["p"] for t in tests] + [t["ps"] for t in tests])
    real = []
    for i, t in enumerate(tests):
        t["pa"], t["psa"] = adj[i], adj[len(tests) + i]
        name = "pool 131K-400K" if t["d"] == "pool" else kname(t["d"])
        lines.append(f"  {name:>14s}  n {t['n']:3d}  exact {t['ea']}/{t['n']} vs {t['eb']}/{t['n']} (diff {(t['ea'] - t['eb']) / t['n']:+.3f})  "
                     f"b {t['b']} c {t['c']}  McNemar p {t['p']:.3g} Holm {t['pa']:.3g} {'REAL' if t['pa'] < 0.05 else 'NOT SHOWN'}  |  "
                     f"score sign {t['pos']}+/{t['neg']}- p {t['ps']:.3g} Holm {t['psa']:.3g} {'REAL' if t['psa'] < 0.05 else 'NOT SHOWN'}")
        if t["pa"] < 0.05:
            real.append(f"{name} exact {t['ea'] - t['eb']:+d}/{t['n']}")
        if t["psa"] < 0.05:
            real.append(f"{name} score {'higher' if t['pos'] > t['neg'] else 'lower'} for {a}")
    if not tests:
        emit(f"lcqpair-{a}-{b}", "INFO (no task both arms answered)")
    else:
        pool = next((t for t in tests if t["d"] == "pool"), None)
        ptxt = (f"pool 131K-400K exact {pool['ea']}/{pool['n']} vs {pool['eb']}/{pool['n']}, b {pool['b']} c {pool['c']}, Holm p {pool['pa']:.3g}; " if pool
                else "no task at 131K-400K both arms answered, so no pooled test; tested depths: " + ", ".join(kname(t["d"]) for t in tests) + "; ")
        emit(f"lcqpair-{a}-{b}", "INFO (" + ptxt + ("real differences: " + "; ".join(real) if real else "no difference shown at Holm 0.05") + ")")
    return lines


def spread_table(arm, rows, emit):
    s = [r for r in rows if r.get("phase") == "samp"]
    if not s:
        return []
    by0 = t0(rows)
    reps = sorted({r["rep"] for r in s})
    lines = [f"{arm}: sampled passes (temperature {s[0].get('temperature')}), reps {', '.join(map(str, reps))}, against temperature 0 on the same tasks"]
    parts = []
    for d in sorted({r["depth"] for r in s}) + ["all"]:
        sd = [r for r in s if d == "all" or r["depth"] == d]
        keys = sorted({(r["item"], r["task"]) for r in sd})
        per = {rp: {(r["item"], r["task"]): r for r in sd if r["rep"] == rp} for rp in reps}
        txt = [f"rep {rp} {fmt_ci(exact_of(per[rp].values()), len(per[rp]))}" for rp in reps]
        txt.append(f"T0 {fmt_ci(sum(1 for k in keys if k in by0 and by0[k].get('exact') is True), sum(1 for k in keys if k in by0))}")
        if len(reps) >= 2:
            r1, r2 = per[reps[0]], per[reps[1]]
            both = [k for k in keys if k in r1 and k in r2]
            agree = sum(1 for k in both if (r1[k].get("exact") is True) == (r2[k].get("exact") is True))
            bb = sum(1 for k in both if r1[k].get("exact") is True and r2[k].get("exact") is not True)
            cc = sum(1 for k in both if r1[k].get("exact") is not True and r2[k].get("exact") is True)
            txt.append(f"agree {agree}/{len(both)}, rep {reps[0]} vs {reps[1]} b {bb} c {cc} McNemar p {binom_two_sided(bb, cc):.3g}")
        lines.append(f"  {'all' if d == 'all' else kname(d):>5s}  " + "; ".join(txt))
        if d == "all":
            parts = txt
    emit(f"lcqspread-{arm}", "INFO (" + "; ".join(parts) + ")")
    return lines


def main(argv):
    if len(argv) < 4 or argv[1] != "table":
        print(USAGE, file=sys.stderr)
        return 2
    out, arms = argv[2], argv[3:]
    try:
        g = os.fdopen(3, "w")
    except OSError:
        g = sys.stderr

    def emit(n, v):
        g.write(f"GATE\t{n}\t{v}\n")
        g.flush()

    rows = {a: load(out, a) for a in arms}
    if not any(rows.values()):
        print("lcqstats: no records for " + " ".join(arms), file=sys.stderr)
        return 2
    print("LCQ v2: exact rate with Wilson 95% intervals; paired McNemar / sign tests, Holm-adjusted; REAL = adjusted p < 0.05\n")
    for a in arms:
        print("\n".join(arm_table(a, rows[a], emit)) + "\n")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            if rows[a] and rows[b]:
                print("\n".join(pair_table(a, rows[a], b, rows[b], emit)) + "\n")
    for a in arms:
        ls = spread_table(a, rows[a], emit)
        if ls:
            print("\n".join(ls) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
