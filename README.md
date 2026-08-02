# kmd

Korean-aware markdown search — a qmd-compatible CLI built on tantivy + lindera (ko-dic).

## Why

[qmd](https://github.com/tobi/qmd)'s BM25 (SQLite FTS5 unicode61 + per-char CJK
normalization) misses most Korean queries. kmd indexes Korean text with proper
morphological analysis ("인증서를" → "인증서" + "를"), so Korean keyword search
actually works.

Measured on 60 real Korean prompts (Claude Code RAG hook simulation):

| Engine | Korean hit rate | English hit rate | hook latency |
|---|---|---|---|
| qmd 1.0 (BM25) | 13% | 52% | ~150ms |
| qmd 2.6.3 (BM25+CJK per-char) | 18% | 67% | ~150ms |
| **kmd (lindera BM25)** | **72%** | 62% | **~82ms (daemon) / ~370ms (cold)** |

## Usage

```sh
kmd update              # index collections from ~/.config/kmd/index.yml
kmd search "인증서 SAN" -n 5 --json
kmd status

# Track runtime, sessions, RAG history, and persisted self-checks.
kmd dashboard
kmd dashboard --json
kmd dashboard --check

# Install Claude Code integration without overwriting other hooks.
kmd hook install
kmd hook status
kmd hook uninstall

# Date → project → session/learning view. The current directory is the locality anchor.
kmd journal
kmd journal --date 2026-08-01
kmd journal --project platform-tools
kmd journal --cwd ~/src/sendbird/soda-k8s/.worktree/cohome --days 14
kmd journal --json
```

`kmd dashboard` provides Overview, Activity, Journal, RAG, Evaluations, and Checks
tabs. Use `←`/`→` or `1`–`6` to switch tabs, `↑`/`↓` to scroll, `r` to refresh, and
`t` to run self-checks. The latest 500 check results are kept in
`~/.local/state/kmd/checks.jsonl`. Successful `eval`, `util`, and `ab` runs are kept
in `~/.local/state/kmd/evaluations.jsonl`. The same state is available
non-interactively through `--json`; `--check` validates config, store, Tantivy,
daemon, Claude hooks, RAG/evaluation logs, search, activity, and journal.

`kmd hook install` idempotently adds `SessionStart`, `UserPromptSubmit`, `Stop`,
`SessionEnd`, and write-oriented `PostToolUse` entries to `~/.claude/settings.json`.
It resolves the currently running kmd binary, preserves unrelated settings and hooks,
and `kmd hook uninstall` removes only kmd-owned commands.

`kmd journal` combines historical learnings with live session activity. Entries are
ordered by date and then by spatial locality: exact cwd, same worktree, same canonical
Git repository, and touched-file overlap. New learnings persist their creation `Cwd:`;
older learnings recover it from the original transcript or infer it from touched files.

Reads qmd's `index.yml` as-is; emits qmd-compatible `--json` output
(docid/score/file/title/context/snippet), so it drops into scripts that
already consume `qmd search --json`.

## Layout

- `src/config.rs` — index.yml loading (qmd-compatible)
- `src/store.rs` — SQLite document store (source of truth, dirty tracking)
- `src/scan.rs` — collection scan, incremental upsert by content hash
- `src/tokenize.rs` — lindera ko-dic (Korean) + lowercase/stemmer (English)
- `src/bm25.rs` — tantivy index, dual ko/en fields, search + snippets
- `src/output.rs` — qmd-compatible JSON / CLI output
- `src/dashboard.rs` — operations TUI, JSON snapshot, checks and evaluation history
- `src/evaluation_log.rs` — bounded history for successful L1/L2/L3 evaluation runs
- `src/eval.rs` — L1 retrieval eval (`kmd eval`): known-item + gold
- `src/util.rs` — L2 injection utilization (`kmd util`)
- `src/ab.rs` — L3 A/B blind comparison (`kmd ab`)

## Roadmap

- [x] vector search (embeddinggemma GGUF via llama-cpp-2, `embed` feature)
- [x] hybrid RRF (`kmd query`)
- [x] warm daemon (unix socket) for sub-100ms hook latency

## Evaluation

3-layer utility eval — see [`eval/README.md`](eval/README.md).

| Layer | Question | Command |
|---|---|---|
| L1 retrieval | found the right doc? | `kmd eval --compare-qmd` |
| L2 utilization | did the answer use it? | `kmd util` |
| L3 A/B | was the answer better? | `kmd ab --prompts eval/prompts.txt` |

Measured (known-item, 341 queries, k=5): Korean R@5 **kmd 69% vs qmd 44%**,
English **kmd 99% vs qmd 85%**. Blind A/B (10 pairs): **kmd 70% win**.
Reproduce with `make eval-all`.
