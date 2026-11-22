#!/usr/bin/env python3
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

USAGE = """fetch_corpus.py [--cache DIR] [--out FILE.jsonl.gz] [--delay SECONDS] [--offline]

Rebuilds corpus/corpus.jsonl.gz, the filler text of every haystack, from public-domain books on Project Gutenberg.

corpus/books.tsv pins each book: ebook number, title, author, the exact text-file URL, its sha256 and size. Each file is
downloaded once into --cache (default corpus/cache), one file at a time with --delay (5) seconds between downloads, with
a generic User-Agent, and must match its sha256 (a file Project Gutenberg has since changed stops the script and is
named). When the pinned URL fails, the same file at www.gutenberg.org/cache/epub/N/pgN.txt is tried, against the same
sha256. --offline uses the cache only.

The text between the "*** START OF THE PROJECT GUTENBERG EBOOK" and "*** END OF THE PROJECT GUTENBERG EBOOK" lines is
kept (the licence and trademark text outside them is dropped), line endings become \\n, trailing spaces go, and a fixed
table maps typographic characters to ASCII (curly quotes, dashes, ellipsis, accented Latin letters). corpus/blocks.tsv
names every block: its book, first and last line of that kept text, its frame (file: a read_file tool result; note:
assistant notes), its token count and its sha256. Each block is rebuilt from those lines and must match its sha256;
the corpus is one JSON line a block, {"k": frame, "t": text, "n": tokens}, gzipped with no name and mtime 0.
The uncompressed corpus's sha256 is checked against corpus/SHA256SUMS; the gzip bytes are reported (zlib builds can
differ there, the uncompressed text is the reference). Standard library only. Exit 0 when everything matches, 1 when
something differs, 2 on bad arguments."""

HERE = os.path.dirname(os.path.realpath(__file__))
CDIR = os.path.join(HERE, "corpus")
UA = "bench-corpus-fetch/1.0 (public-domain text for a reproducible benchmark)"
FALLBACK = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)

ASCII = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    "–": "-", "‐": "-", "‑": "-", "‒": "-", "—": "--", "―": "--", "−": "-",
    "…": "...", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "·": ".", "•": "*", "×": "x", "½": "1/2", "¼": "1/4", "¾": "3/4", "°": " degrees",
    "£": "L", "§": "S.", "¶": "P.", "†": "+", "‡": "++", "﻿": "",
    "À": "A", "Á": "A", "Â": "A", "Ã": "A", "Ä": "A", "Å": "A", "Æ": "AE", "Ç": "C",
    "È": "E", "É": "E", "Ê": "E", "Ë": "E", "Ì": "I", "Í": "I", "Î": "I", "Ï": "I",
    "Ñ": "N", "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O", "Ö": "O", "Ø": "O", "Ù": "U",
    "Ú": "U", "Û": "U", "Ü": "U", "Ý": "Y", "ß": "ss",
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a", "æ": "ae", "ç": "c",
    "è": "e", "é": "e", "ê": "e", "ë": "e", "ì": "i", "í": "i", "î": "i", "ï": "i",
    "ñ": "n", "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ø": "o", "ù": "u",
    "ú": "u", "û": "u", "ü": "u", "ý": "y", "ÿ": "y", "Œ": "OE", "œ": "oe",
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
}
ASCII_TABLE = str.maketrans(ASCII)


def read_tsv(path):
    with open(path, encoding="ascii") as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def books():
    return read_tsv(os.path.join(CDIR, "books.tsv"))


def blocks():
    return read_tsv(os.path.join(CDIR, "blocks.tsv"))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 200:
            raise urllib.error.HTTPError(url, r.status, "not 200", r.headers, None)
        return r.read()


def fetch(book, cache, delay, offline, state):
    path = os.path.join(cache, f"{book['ebook']}.txt")
    if os.path.exists(path):
        b = open(path, "rb").read()
        if sha(b) == book["sha256"]:
            return b, "cache"
    if offline:
        raise SystemExit(f"fetch_corpus: ebook {book['ebook']} is not in {cache} with sha256 {book['sha256'][:12]} (--offline)")
    errs = []
    for url in (book["url"], FALLBACK.format(n=book["ebook"])):
        if state["last"]:
            time.sleep(max(0.0, delay - (time.time() - state["last"])))
        try:
            b = download(url)
        except (urllib.error.URLError, OSError) as e:
            errs.append(f"{url}: {e}")
            continue
        finally:
            state["last"] = time.time()
        if sha(b) != book["sha256"]:
            errs.append(f"{url}: sha256 {sha(b)[:12]}, {len(b)} bytes, not {book['sha256'][:12]}, {book['bytes']} bytes (the file changed upstream)")
            continue
        os.makedirs(cache, exist_ok=True)
        with open(path + ".part", "wb") as f:
            f.write(b)
        os.replace(path + ".part", path)
        return b, url
    raise SystemExit(f"fetch_corpus: ebook {book['ebook']} ({book['title']}): " + "; ".join(errs))


def body_lines(raw):
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    a = next((i for i, l in enumerate(lines) if START.match(l)), None)
    b = next((i for i, l in enumerate(lines) if END.match(l)), None)
    if a is None or b is None or b <= a:
        raise ValueError("no START/END markers")
    return [l.rstrip().translate(ASCII_TABLE).rstrip() for l in lines[a + 1:b]]


def frame(kind, book, first, last, body):
    if kind == "file":
        return f"tool: read_file path=books/{book['slug']}.txt lines={first}-{last}\n```text\n{body}\n```\n\n"
    return f"assistant: Notes I took from {book['title']} (lines {first}-{last}):\n{body}\n\n"


def block_text(kind, book, lines, first, last):
    return frame(kind, book, first, last, "\n".join(lines[first - 1:last]))


def write_corpus(rows, out):
    raw = "".join(json.dumps(r) + "\n" for r in rows).encode("ascii")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as g:
        g.write(raw)
    gz = buf.getvalue()
    if out:
        with open(out + ".part", "wb") as f:
            f.write(gz)
        os.replace(out + ".part", out)
    return sha(raw), sha(gz), len(raw), len(gz)


def sums():
    p = os.path.join(CDIR, "SHA256SUMS")
    out = {}
    if os.path.exists(p):
        for line in open(p):
            h, _, name = line.strip().partition("  ")
            out[name] = h
    return out


def load_bodies(cache, delay, offline):
    state = {"last": 0.0}
    out = {}
    for bk in books():
        raw, src = fetch(bk, cache, delay, offline, state)
        out[bk["ebook"]] = (bk, body_lines(raw))
        print(f"ebook {bk['ebook']:>6} {bk['title'][:40]:40s} {len(raw):>8} bytes  sha256 ok  ({'cache' if src == 'cache' else 'downloaded'})", file=sys.stderr, flush=True)
    return out


def main(argv):
    args = {"--cache": os.path.join(CDIR, "cache"), "--out": os.path.join(CDIR, "corpus.jsonl.gz"), "--delay": "5"}
    offline = False
    i = 1
    while i < len(argv):
        if argv[i] in ("-h", "--help"):
            print(USAGE)
            return 0
        if argv[i] == "--offline":
            offline = True
            i += 1
            continue
        if argv[i] in args and i + 1 < len(argv):
            args[argv[i]] = argv[i + 1]
            i += 2
            continue
        print(USAGE, file=sys.stderr)
        return 2
    bodies = load_bodies(args["--cache"], float(args["--delay"]), offline)
    rows, bad = [], []
    for r in blocks():
        bk, lines = bodies[r["ebook"]]
        t = block_text(r["kind"], bk, lines, int(r["first"]), int(r["last"]))
        if not t.isascii() or sha(t.encode("ascii", "replace")) != r["sha256"]:
            bad.append(f"block {r['idx']} (ebook {r['ebook']} lines {r['first']}-{r['last']})")
        rows.append({"k": r["kind"], "t": t, "n": int(r["tokens"])})
    if bad:
        print(f"fetch_corpus: {len(bad)} of {len(rows)} blocks differ from blocks.tsv: {'; '.join(bad[:5])}", file=sys.stderr)
        return 1
    h, hg, n, ng = write_corpus(rows, args["--out"])
    want = sums().get("corpus.jsonl")
    print(f"corpus: {len(rows)} blocks from {len(bodies)} books, {sum(r['n'] for r in rows)} tokens; corpus.jsonl {n} bytes sha256 {h}; "
          f"{os.path.basename(args['--out'])} {ng} bytes sha256 {hg}")
    if want and want != h:
        print(f"fetch_corpus: corpus.jsonl sha256 {h} is not SHA256SUMS's {want}", file=sys.stderr)
        return 1
    wg = sums().get("corpus.jsonl.gz")
    if wg and wg != hg:
        print(f"note: the gzip bytes differ from SHA256SUMS's {wg[:12]} (another zlib build); the uncompressed corpus matches", file=sys.stderr)
    print("match: every book's sha256, every block's sha256 and the corpus's sha256" if want else "no SHA256SUMS entry for corpus.jsonl to compare")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
