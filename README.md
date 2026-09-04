# kmd

Korean-aware markdown search over notes, past Claude Code sessions, and project source.
Built on tantivy + lindera (ko-dic).

## Why

kmd replaced [qmd](https://github.com/tobi/qmd), whose BM25 (SQLite FTS5 unicode61 +
per-char CJK normalization) missed most Korean queries. kmd indexes Korean text with
proper morphological analysis ("인증서를" → "인증서" + "를"), so Korean keyword search
actually works.

Measured on 60 real Korean prompts (Claude Code RAG hook simulation) while both
engines were installed side by side:

| Engine | Korean hit rate | English hit rate | hook latency |
|---|---|---|---|
| qmd 1.0 (BM25) | 13% | 52% | ~150ms |
| qmd 2.6.3 (BM25+CJK per-char) | 18% | 67% | ~150ms |
| **kmd (lindera BM25)** | **72%** | 62% | **~82ms (daemon) / ~370ms (cold)** |

qmd has since been removed from this environment; the comparison numbers above are
kept as the historical record that motivated kmd, and are no longer reproducible
locally.

## Usage

```sh
kmd update              # index collections from ~/.config/kmd/index.yml
kmd search "인증서 SAN" -n 5 --json
kmd status

# Cross-axis search: past sessions, curated knowledge, and project source at once.
kmd global "worktree"
kmd global "SearchSessions" --axis project
kmd global "인증서 갱신" -n 3 --json
```

### Axes

`kmd search` queries one collection at a time. `kmd global` queries three *axes* and
returns a quota of hits from each, so no single axis crowds out the rest:

| Axis | Contents |
|---|---|
| `session` | Past Claude Code sessions (`learnings`, `narwhal-runs`) |
| `knowledge` | Curated notes — wiki, memory, rules, skills, contexts |
| `project` | Source code across local git repositories |

A single ranked list would not work here: code repeats identifiers so its term
frequencies run high, while session summaries are short. Merged into one list, one axis
takes every top slot. Per-axis quotas keep all three visible.

An axis is assigned per collection in `index.yml` via `axis:`. Collections without it
fall back to a name-based default (`learnings` and `narwhal-runs` → `session`, everything
else → `knowledge`), so existing configs keep working untouched.

The `project` axis is deliberately excluded from the RAG hook's automatic injection
(`rag.rs`'s `CLAUDE_COLLECTIONS`) — source files surface only when you ask for them.

### Indexing source code

Code collections use `source: git-repos`, which finds git repositories under `path` and
enumerates each one with `git ls-files`. Build artifacts drop out via `.gitignore` for
free, and it beats a full directory walk (203 repos enumerate in ~6s).

```yaml
  project:
    path: /Users/me/src
    source: git-repos
    axis: project
    store_body: false
    max_file_size: 1048576
    exclude:
      - "**/vendor/**"
      - "**/*.pb.go"
```

`store_body: false` (the default for `source: git-repos`) records only the absolute path
instead of copying file contents into the store. Indexing and snippets read the original
file. Source trees are large and stay put on disk, so a second copy would waste space and
go stale; notes collections keep `store_body: true` because their contents are the
document.

```sh
# Everything else is unchanged.
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

`kmd dashboard` uses three focused views instead of separate tabs for each data source:

| View | Function |
|---|---|
| Sessions | Live Claude sessions merged with the kmd queries and retrieval results of each session |
| Operations | Runtime, retrieval, evaluation, and self-check health in one scrollable view |
| Simulator | Real RAG pipeline, raw BM25, or cross-axis Global queries against the local index |

Sessions shows which Claude sessions are alive right now. Liveness is derived from the
data kmd already owns: the `UserPromptSubmit` hook only runs inside a running session, so
the most recent `rag.jsonl` query timestamp proves that session was alive at that moment.
Rows are tagged `active` (queried within 15 minutes), `recent` (within an hour), or `idle`.

Each row also records where its data came from — `rag`, `live`, or `live+rag`:

- `rag` — `searched` entries from the last seven days of `rag.jsonl`: the extracted query,
  latency/injection metadata, and each hit's path, score, and snippet.
- `live` — activity cards for Claude sessions seen in the last 24 hours: repo, cwd, and the
  latest request/result summary that the `UserPromptSubmit` hook already shares across sessions.

Neither source reads or displays full Claude conversation transcripts.

The same `active` / `recent` / `idle` tag is attached to the `<daily-activity>` block that
`kmd rag --hook` injects on every prompt, so a running session can tell which of its peers
are working right now rather than which ones merely finished something today.

### Keys

No binding uses `Ctrl`, `Alt`, or `Cmd` — those chords stay with the terminal and
the tmux prefix, and the dashboard passes them through untouched. Press `?` for
the full keymap.

| Key | Action |
|---|---|
| `1`–`3`, `Tab`/`Shift-Tab`, `h`/`l`, `←`/`→` | switch view |
| `j`/`k`, `↑`/`↓` | select a query session or scroll Operations |
| `PgUp`/`PgDn`, `g`/`G` | scroll query results or the current view |
| `r` | refresh the snapshot |
| `t` | run the self-check suite |
| `?` | toggle the keymap overlay |
| `q`, `Esc` | quit |

The Simulator view splits input into command and typing modes:

| Mode | Key | Action |
|---|---|---|
| command | `i` or `/` | start typing a query |
| command | `Enter` | run the query |
| command | `m` | RAG pipeline ⇄ BM25 search ⇄ Global |
| command | `j`/`k` | select a hit |
| command | `J`/`K` | scroll result detail |
| command | `n`/`p` | next / previous prompt history |
| command | `x` | clear the query |
| typing | `Esc` | back to command mode |
| typing | `Enter` | run the query |
| typing | `↑`/`↓` | recall prompt history |

RAG mode shows gate → extracted query → filtered hits → injected context; BM25 mode
shows raw retrieval; Global mode is `kmd global` in the TUI — hits arrive grouped under
`── session` / `── knowledge` / `── project` headers, each axis holding its own quota, and
an axis with no hits keeps its header so "nothing matched" stays distinguishable from
"that axis is not indexed." Selection still walks the hits themselves, skipping headers.
Simulator runs never write `rag.jsonl`. Semantic colors remain readable through text and
symbols; set `NO_COLOR=1` to disable hues.

The latest 500 check results are kept in `~/.local/state/kmd/checks.jsonl`.
Successful `eval`, `util`, and `ab` runs are kept in
`~/.local/state/kmd/evaluations.jsonl`. The same operational state is available
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

`--json` output keeps the docid/score/file/title/context/snippet shape inherited
from qmd, so scripts written against the old format still work, plus an `abspath`
field: a hit's `file` is a `kmd://<collection>/<relpath>` URI, and `abspath` is
the real path to open. It is absent only when the hit's collection is no longer
in `index.yml`, in which case no path can be derived and saying so beats guessing.

## MCP (other clients)

Claude Code reaches kmd two ways already — the `UserPromptSubmit` hook injects
context automatically, and the `kmd` skill runs the CLI. Neither is available in
Claude Desktop, Cursor, or Codex, which speak MCP instead.

```sh
claude mcp add --scope user kmd kmd mcp   # or the equivalent for your client
```

`kmd mcp` speaks JSON-RPC over stdio and exposes two tools:

| Tool | Purpose |
|---|---|
| `kmd_search` | Cross-axis search; `axis` narrows to session/knowledge/project |
| `kmd_status` | What the index holds — tells an empty index from a missed query |

Search goes through the daemon when one is running, so the warm Tantivy reader
is reused rather than reopened per client-spawned process; it falls back
in-process otherwise. Results are rendered as text, not JSON — an MCP response
lands verbatim in the model's context, and JSON punctuation costs tokens without
helping it read. Each hit carries `abspath` so the client can open the file.

Do not add this to Claude Code on top of the hook and skill. With this many MCP
servers registered, tools arrive deferred: using one costs a `ToolSearch` round
trip first, whereas `Bash(kmd:*)` is callable immediately.

## Layout

- `src/config.rs` — index.yml loading, collection sources and axes
- `src/store.rs` — SQLite document store (source of truth, dirty tracking)
- `src/scan.rs` — collection scan, incremental upsert by content hash
- `src/tokenize.rs` — lindera ko-dic (Korean) + lowercase/stemmer (English)
- `src/bm25.rs` — tantivy index, dual ko/en fields, search + snippets
- `src/global.rs` — cross-axis search (`kmd global`)
- `src/mcp.rs` — MCP server over stdio (`kmd mcp`), for non-Claude-Code clients
- `src/output.rs` — JSON / CLI output, per-axis rendering
- `src/dashboard.rs` — operations TUI, JSON snapshot, checks and evaluation history
- `src/evaluation_log.rs` — bounded history for successful L1/L2/L3 evaluation runs
- `src/eval.rs` — L1 retrieval eval (`kmd eval`): known-item + gold
- `src/util.rs` — L2 injection utilization (`kmd util`)

## Roadmap

- [x] vector search (embeddinggemma GGUF via llama-cpp-2, `embed` feature)
- [x] hybrid RRF (`kmd query`)
- [x] warm daemon (unix socket) for sub-100ms hook latency

## Evaluation

Two-layer utility eval — see [`eval/README.md`](eval/README.md).

| Layer | Question | Command |
|---|---|---|
| L1 retrieval | found the right doc? | `kmd eval` |
| L2 utilization | did the answer use it? | `kmd util` |

Current (known-item, 400 queries, k=5): R@5 **81%** overall — Korean **75%**,
English **97%**. Reproduce with `make eval-all`.

The L3 A/B layer compared kmd against qmd head-to-head; it was removed along with
qmd itself.
