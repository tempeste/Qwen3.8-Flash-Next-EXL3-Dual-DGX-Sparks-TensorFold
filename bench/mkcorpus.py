#!/usr/bin/env python3
import hashlib
import importlib.util
import os
import re
import sys
import time

USAGE = """mkcorpus.py: how corpus/books.tsv and corpus/blocks.tsv were made (fetch_corpus.py rebuilds the corpus from them).

  mkcorpus.py pin BOOKS.txt [--cache DIR] [--delay S]
      BOOKS.txt: one Project Gutenberg ebook number a line. Downloads each pgN.txt from gutenberg.pglaf.org (PGLAF's
      own mirror), one at a time, and writes corpus/books.tsv: ebook, title and author (from the file's header), slug,
      URL, sha256 and size. A book is refused unless its header says English and carries the US public-domain line
      ("for the use of anyone anywhere in the United States"), and no copyright notice.
  mkcorpus.py cut TOKENIZER.json [--target 1000000] [--note-share 0.35] [--drop PATTERNS.txt] [--cache DIR]
      Cuts each book's kept text (fetch_corpus.body_lines) into blocks of 24 to 72 lines (sizes from splitmix64 keyed
      by the ebook number), blank lines trimmed at both ends. Each book's blocks are visited in bit-reversed order, so
      the first k picks of a book spread over the whole book; books take turns in books.tsv order. A block is kept
      when its text is ASCII after the character table, at least 200 characters, holds no "```", no word "Gutenberg",
      nothing shaped like an LCQ needle id, and matches no regex of PATTERNS.txt. Its frame is "note" while note
      tokens are below note-share of all tokens, else "file". Tokens are counted with TOKENIZER.json (the model's;
      needs the tokenizers package) on the framed text, until target. Writes corpus/blocks.tsv, the corpus and
      corpus/SHA256SUMS."""

HERE = os.path.dirname(os.path.realpath(__file__))
spec = importlib.util.spec_from_file_location("fetch_corpus", os.path.join(HERE, "fetch_corpus.py"))
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
MIRROR = "https://gutenberg.pglaf.org/cache/epub/{n}/pg{n}.txt"
NEEDLE = re.compile(r"\b(SVC|PRJ|CC|PSG|EMP|E)-[0-9A-Z]{2,}")
M64 = 0xFFFFFFFFFFFFFFFF


def mix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def field(head, name):
    out = None
    for i, l in enumerate(head):
        if l.startswith(name + ":"):
            out = l[len(name) + 1:].strip()
            j = i + 1
            while j < len(head) and head[j].startswith(" ") and head[j].strip():
                out += " " + head[j].strip()
                j += 1
            break
    return out


def pin(listing, cache, delay):
    nums = [l.split("#")[0].strip() for l in open(listing)]
    nums = [n for n in nums if n]
    if len(set(nums)) != len(nums):
        raise SystemExit("mkcorpus: an ebook number is listed twice")
    rows, refused, last = [], [], 0.0
    os.makedirs(cache, exist_ok=True)
    for n in nums:
        path = os.path.join(cache, f"{n}.txt")
        url = MIRROR.format(n=n)
        if os.path.exists(path):
            b = open(path, "rb").read()
        else:
            if last:
                time.sleep(max(0.0, delay - (time.time() - last)))
            try:
                b = F.download(url)
            except OSError as e:
                last = time.time()
                refused.append(f"{n}: {url}: {e}")
                print(f"refused {n}: {url}: {e}", file=sys.stderr, flush=True)
                continue
            last = time.time()
            with open(path, "wb") as f:
                f.write(b)
        text = b.decode("utf-8-sig").replace("\r\n", "\n")
        head = text.split("\n")
        a = next((i for i, l in enumerate(head) if F.START.match(l)), None)
        head = head[:a] if a is not None else []
        hs = "\n".join(head)
        title = (field(head, "Title") or "").translate(F.ASCII_TABLE)
        author = (field(head, "Author") or field(head, "Editor") or "Anonymous").translate(F.ASCII_TABLE)
        why = []
        if a is None:
            why.append("no START marker")
        if field(head, "Language") != "English":
            why.append(f"language {field(head, 'Language')}")
        if "for the use of anyone anywhere in the United States" not in re.sub(r"\s+", " ", hs):
            why.append("no US public-domain line")
        if re.search(r"copyright", hs, re.I):
            why.append("a copyright notice in the header")
        if not title.isascii() or not author.isascii() or "\t" in title + author:
            why.append("title or author not ASCII after the table")
        try:
            F.body_lines(b)
        except ValueError as e:
            why.append(str(e))
        if why:
            refused.append(f"{n} ({title}): {', '.join(why)}")
            print(f"refused {n} {title}: {', '.join(why)}", file=sys.stderr, flush=True)
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48].strip("-")
        rows.append([n, title, author, slug, url, F.sha(b), str(len(b))])
        print(f"pinned {n:>6} {title[:50]:50s} {author[:30]:30s} {len(b):>8} bytes", file=sys.stderr, flush=True)
    with open(os.path.join(F.CDIR, "books.tsv"), "w", encoding="ascii") as f:
        f.write("ebook\ttitle\tauthor\tslug\turl\tsha256\tbytes\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"books.tsv: {len(rows)} books pinned, {len(refused)} refused" + (": " + "; ".join(refused) if refused else ""))
    return 0


def chunks(ebook, lines):
    s = mix(int(ebook))
    out = []
    i = 0
    while i < len(lines):
        s = mix(s)
        size = 24 + s % 49
        a, b = i, min(len(lines), i + size)
        i = b
        while a < b and not lines[a].strip():
            a += 1
        while b > a and not lines[b - 1].strip():
            b -= 1
        if b > a:
            out.append((a + 1, b))
    return out


def bitrev_order(n):
    if n <= 1:
        return list(range(n))
    bits = (n - 1).bit_length()
    return sorted(range(n), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))


def cut(tokp, target, share, drop, cache):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokp)
    count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    pats = [re.compile(l.rstrip("\n")) for l in open(drop) if l.strip()] if drop else []
    bodies = F.load_bodies(cache, 5.0, True)
    queues = []
    for bk in F.books():
        b, lines = bodies[bk["ebook"]]
        cs = chunks(bk["ebook"], lines)
        queues.append([bk, lines, [cs[i] for i in bitrev_order(len(cs))]])
    rows, total, note, dropped = [], 0, 0, {"short": 0, "ascii": 0, "fence": 0, "pg": 0, "needle": 0, "pattern": 0}
    while total < target and queues:
        nxt = []
        for q in queues:
            if total >= target:
                nxt.append(q)
                continue
            bk, lines, cs = q
            while cs:
                first, last = cs.pop(0)
                body = "\n".join(lines[first - 1:last])
                why = ("short" if len(body) < 200 else "ascii" if not body.isascii() else "fence" if "```" in body else
                       "pg" if re.search(r"gutenberg", body, re.I) else "needle" if NEEDLE.search(body) else
                       "pattern" if any(p.search(body) for p in pats) else None)
                if why:
                    dropped[why] += 1
                    continue
                kind = "note" if note < share * max(total, 1) else "file"
                t = F.block_text(kind, bk, lines, first, last)
                if any(p.search(t) for p in pats):
                    dropped["pattern"] += 1
                    continue
                n = count(t)
                rows.append([str(len(rows)), bk["ebook"], str(first), str(last), kind, str(n), F.sha(t.encode("ascii"))])
                total += n
                note += n if kind == "note" else 0
                break
            if cs:
                nxt.append(q)
        queues = nxt
    with open(os.path.join(F.CDIR, "blocks.tsv"), "w", encoding="ascii") as f:
        f.write("idx\tebook\tfirst\tlast\tkind\ttokens\tsha256\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    out = [{"k": r[4], "t": F.block_text(r[4], bodies[r[1]][0], bodies[r[1]][1], int(r[2]), int(r[3])), "n": int(r[5])} for r in rows]
    h, hg, nb, ng = F.write_corpus(out, os.path.join(F.CDIR, "corpus.jsonl.gz"))
    with open(os.path.join(F.CDIR, "SHA256SUMS"), "w") as f:
        f.write(f"{h}  corpus.jsonl\n{hg}  corpus.jsonl.gz\n")
    per = {}
    for r in rows:
        per[r[1]] = per.get(r[1], 0) + int(r[5])
    print(f"blocks.tsv: {len(rows)} blocks, {total} tokens ({note} in notes), from {len(per)} books "
          f"({min(per.values())}-{max(per.values())} tokens a book); left out: {dropped}; corpus.jsonl sha256 {h}, gz {hg}")
    return 0


def main(argv):
    opts = {"--cache": os.path.join(F.CDIR, "cache"), "--delay": "5", "--target": "1000000", "--note-share": "0.35", "--drop": ""}
    pos = []
    i = 2
    while i < len(argv):
        if argv[i] in opts and i + 1 < len(argv):
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    if len(argv) > 1 and argv[1] == "pin" and len(pos) == 1:
        return pin(pos[0], opts["--cache"], float(opts["--delay"]))
    if len(argv) > 1 and argv[1] == "cut" and len(pos) == 1:
        return cut(pos[0], int(opts["--target"]), float(opts["--note-share"]), opts["--drop"], opts["--cache"])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
