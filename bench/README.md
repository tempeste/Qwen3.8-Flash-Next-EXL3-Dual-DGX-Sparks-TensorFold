# bench: prompt-reuse tests

The tables in the top-level README report three serving features that a lone-prompt benchmark cannot show:
prompt-cache reuse (a new question on a document already read), shared-prefix keep (requests that share an opening
and then part), and prompt-cache size (how much read text a server still holds later). This folder has the tests that
measured them, so anyone can rerun them on any OpenAI-compatible server. They need Python 3.9+ and nothing else.

These tests are **not sparkDash**. sparkDash (MiaAI-Lab's benchmark, which the decode and prompt-reading rows use)
sends unique prompts one at a time, so a cache never gets a second look at the same text. These tests send the same
bytes more than once on purpose and read the server's `usage.prompt_tokens_details.cached_tokens` and first-token
times.

## What each test measures

All prompts come from LCQ v2's haystacks (`lcqgen.py`, planned in `manifest.json`). A haystack is a session line
with a hash, then filler framed as an agent's tool results and notes, with needles placed in it (record lookups among
near-duplicates, two-record joins, passages to copy). Every request is streamed, temperature 0, thinking off
(`chat_template_kwargs.enable_thinking: false`), one at a time.

| test | command | what it sends | what it reports |
|---|---|---|---|
| warm reuse: same document, new question | `reuseprobe.py warm` | the 131,072- and 262,144-token haystacks `h131072-0` and `h262144-0`: one lookup cold, then 4 other questions on the same haystack | warm requests at 100K+ tokens that resumed (cached >= 90% of the prompt), warm and cold first-token medians |
| shared opening, parted early | `reuseprobe.py fork` | 2 sets of 6 workers; a set shares the first 70% of `h32768-<set>` (22,825 tokens), then each worker adds a worker line and its own body (a third of `h8192-0` or `h8192-1`, about 2,680 tokens) and asks for `OK` | workers 3-6 of each set that resumed the shared opening (cached >= 90% of it), cached-token and first-token medians. Worker 1 reads cold; worker 2 parts from worker 1; workers 3+ find the opening only when the server kept a state at the point where worker 2 parted |
| cache retention | `reuseprobe.py ret` | pass 1 reads 9 long haystacks (`h400000-0..3`, `h330000-0..3`, `h262144-0`, 3.18M tokens) with one lookup each; pass 2 asks the 4 oldest a second question | pass-2 documents that came back warm (cached >= 90% of the document) and their first-token median; numeric cache fields of `/health` before and after, when the server has that route |
| long-context accuracy (LCQ v2) | `lcq.py run` | all 28 planned haystacks, 8,192 to 400,000 tokens: 16 questions each, plus two sampled passes (temperature 0.7) on 4 of them | exact rate and score by depth (`lcqstats.py table` adds Wilson intervals and paired tests between servers) |

A server without the feature shows it directly: a server that keeps no state where requests part resumes none of
`fork`'s workers 3-6, and one whose cache is too small for pass 1 gets none of `ret`'s 4 back warm. To measure a feature, run the probe against the server with and without it, each from a
fresh start, at least 3 runs a side, interleaved (A B B A A B ...), and compare the counts and first-token medians.

## Commands

```sh
cd bench
python3 lcqgen.py verify
python3 reuseprobe.py warm out/A1.warm.jsonl --url http://127.0.0.1:8000 --model MODEL --label A1
python3 reuseprobe.py fork out/A1.fork.jsonl --url http://127.0.0.1:8000 --model MODEL --label A1
python3 reuseprobe.py ret  out/A1.ret.jsonl  --url http://127.0.0.1:8000 --model MODEL --label A1
python3 reuseprobe.py report out/*.jsonl
python3 lcq.py run out/A1.lcq.jsonl --url http://127.0.0.1:8000 --model MODEL --label A1
python3 lcqstats.py table out A1 B1
```

`--url` is the server's base URL (the scripts add `/v1/chat/completions`); `--model` is the model name the server
serves. `lcqgen.py verify` rebuilds every planned prompt and checks it against the plan's sha256, and checks that each
question has exactly one right answer in its text; run it once on a new machine before sending anything. `lcqgen.py
show manifest.json h131072-0 look1` prints one prompt.

Each script prints its options with `-h` or no arguments. Useful environment settings: `RP_TIMEOUT` (seconds a
request, 1800), `LCQ_DEPTHS` and `LCQ_HAYS` (which haystacks `warm` and `lcq.py` use), `LCQ_WARM_TASKS`,
`RP_FORK_SETS`, `RP_FORK_WORKERS`, `RP_RET_ITEMS`, `RP_RET_BACK`.

## Runtime

Most of the time goes to reading prompts cold, so it scales with the server's prompt-reading speed. Cold tokens a run,
and the time at a steady 3,000 tokens a second (most servers read long prompts slower than short ones, so take
these as lower bounds):

| test | requests | tokens read cold (a server with no reuse at all) | at 3,000 tok/s |
|---|---:|---:|---:|
| warm | 10 | 1.96M (0.39M with reuse) | 11 min (2 min) |
| fork | 12 | 0.31M (0.08M with reuse) | 2 min (under 1 min) |
| ret | 13 | 4.78M (3.18M when the 4 come back warm) | 27 min (18 min) |
| lcq.py run (all 28 haystacks) | 448, and 128 sampled | 129M (5.69M with reuse) | 12 h (32 min) |

Replies are short (8 to 48 tokens; 320 for the copy
tasks of `lcq.py`).

## Results

Each request is one JSON line in the output file: the probe, set, worker, pass or task, `prompt_tokens`,
`cached_tokens`, `ttft_s` (time to the first token), `total_s`, `completion_tokens`, `finish`, the reply's first
characters, and an `error` field when the request failed. `reuseprobe.py report FILE...` prints one line a file, for
example `same document, new question: 8 of 8 warm requests at 100k+ resumed (cached >= 90%); warm first token median
0.42 s, cold 61.30 s; 0 errors`. The kind of a file comes from its name (`.warm.jsonl`, `.fork.jsonl`, `.ret.jsonl`).

A server that does not report `cached_tokens` cannot be scored on resumes; its first-token times still show reuse
(a resumed 262K prompt answers in well under a second, a cold one takes as long as reading 262K tokens).

## The filler corpus

`corpus/corpus.jsonl.gz` holds 1,599 blocks, 1,000,612 tokens (counted with the model's tokenizer), cut from 64
public-domain English books from Project Gutenberg (`corpus/books.tsv`: title, author, ebook number, the exact file
URL and its sha256). Each block is 24 to 72 lines of one book, framed as a `read_file` tool result (`file`) or as an
assistant's notes (`note`, about 35% of the tokens); `corpus/blocks.tsv` gives the book and line range of every block.

To rebuild it from the books:

```sh
python3 fetch_corpus.py              # downloads each book once into corpus/cache, one file every 5 s
```

`fetch_corpus.py` checks every file's sha256, keeps the text between Project Gutenberg's `*** START OF` and
`*** END OF` lines, maps typographic characters to ASCII, cuts the listed blocks and checks each one's sha256 and the
corpus's (`corpus/SHA256SUMS`). The rebuild is byte-identical; a book that has changed on Project Gutenberg since it
was pinned stops the script with its name. `mkcorpus.py` shows how the books were pinned and the blocks chosen (it
needs the `tokenizers` package and the model's `tokenizer.json`), and `lcqgen.py plan` how `manifest.json` was made.
The licence of the texts is in [`CORPUS-LICENSE`](CORPUS-LICENSE).

The needles, seeds (8181, splitmix64), question texts, grading and every probe's method are the ones behind the
README's tables. Only the filler is new: the README's results used filler cut from a private source repository, so
the haystacks here are different bytes at the same depths, and a rerun gives comparable, not identical, numbers.
