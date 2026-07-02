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
kmd update              # index collections from ~/.config/qmd/index.yml
kmd search "인증서 SAN" -n 5 --json
kmd status
```

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

## Roadmap

- [ ] vector search (embeddinggemma GGUF via llama-cpp-2, `embed` feature)
- [ ] hybrid RRF (`kmd query`)
- [ ] warm daemon (unix socket) for sub-100ms hook latency
