# Mouseion

Mouseion is a local-first personal knowledge base. It runs as a single Python daemon, stores documents and chunks in SQLite with `sqlite-vector` and FTS5, embeds chunks through local Ollama, and exposes retrieval/admin operations through HTTP MCP tools plus a small web UI.

## Name

The name `Mouseion` refers to the Mouseion of Alexandria, an ancient institution associated with the Library of Alexandria. A mouseion was originally a place dedicated to the Muses, then came to describe centers of learning such as Plato's Academy and Aristotle's Lyceum. The Alexandrian Mouseion is remembered as an effort to gather leading scholars and collect the books known at the time.

This project borrows the name for the same basic idea at a personal scale: a local place to collect, organize, relate, and retrieve knowledge.

## Runtime

- Python 3.13
- uv
- SQLite via APSW, with `sqlite-vector` for vector search
- Ollama running locally with `nomic-embed-text`

Install dependencies:

```bash
uv sync
```

Start the daemon:

```bash
uv run mouseion serve
```

By default this starts `ollama serve` if needed and pulls `EMBEDDING_MODEL` when it is missing. If you run Ollama yourself:

```bash
uv run mouseion serve --no-ollama
```

The web UI is served at `http://127.0.0.1:7778/`. MCP Streamable HTTP is mounted at `http://127.0.0.1:7778/mcp`.

Delete the SQLite database files after an explicit confirmation prompt:

```bash
uv run mouseion nuke-db
```

This removes `mouseion.db`, `mouseion.db-wal`, and `mouseion.db-shm` only. Stop the daemon first.

Show whether the daemon is running and print summary stats:

```bash
uv run mouseion status
```

Recompute `similar_to` graph edges after bulk imports:

```bash
uv run mouseion recompute-edges
```

Manage the vector backend:

```bash
uv run mouseion vector status
uv run mouseion vector mode exact
uv run mouseion vector mode quantized --qbits 4
uv run mouseion vector quantize --qbits 4 --preload
uv run mouseion vector cleanup
```

Exact full-scan vector search is the default. TurboQuant search is opt-in with `qbits` set to `2`, `3`, or `4`.

## Configuration

Settings are loaded from `.env` and environment variables:

| Variable                       | Default                  |
| ------------------------------ | ------------------------ |
| `MOUSEION_HOST`                | `127.0.0.1`              |
| `MOUSEION_PORT`                | `7778`                   |
| `MOUSEION_DATA_DIR`            | `./data`                 |
| `MOUSEION_REPOS_DIR`           | `./repos`                |
| `MOUSEION_MCP_DESCRIPTION`     | unset                    |
| `OLLAMA_HOST`                  | `http://127.0.0.1:11434` |
| `EMBEDDING_MODEL`              | `nomic-embed-text`       |
| `CHUNK_TARGET_TOKENS`          | `512`                    |
| `CHUNK_MAX_TOKENS`             | `1024`                   |
| `CHUNK_MIN_TOKENS`             | `100`                    |
| `SIMILARITY_THRESHOLD`         | `0.82`                   |
| `SIMILARITY_TOP_K`             | `10`                     |
| `SIMILAR_EDGE_RECOMPUTE_HOURS` | `24`                     |
| `RRF_K`                        | `60`                     |
| `MOUSEION_VECTOR_SEARCH_MODE`   | `exact`                  |
| `MOUSEION_VECTOR_QUANTIZATION_QBITS` | `4`                |
| `MOUSEION_VECTOR_QUANTIZE_PRELOAD` | `false`              |
| `MOUSEION_VECTOR_QUANTIZE_MAX_MEMORY` | `30MB`            |
| `LOG_LEVEL`                    | `INFO`                   |

`data/` and cloned repos under `repos/` are intentionally gitignored. `data/mouseion.db` is the source of truth for mouseion content; include `mouseion.db`, `mouseion.db-wal`, and `mouseion.db-shm` when making a file-level backup. Use `mouseion_export` for a markdown dump.

## MCP Tools

- `mouseion_add_url`
- `mouseion_add_file`
- `mouseion_add_memory`
- `mouseion_add_repo`
- `mouseion_search`
- `mouseion_get_document`
- `mouseion_list`
- `mouseion_relate`
- `mouseion_delete`
- `mouseion_recompute_edges`
- `mouseion_export`

All tools return JSON. The daemon does retrieval only; answer synthesis belongs to the calling AI harness.

## Development

```bash
uv run ruff check .
uv run pyright
uv run pytest
```
