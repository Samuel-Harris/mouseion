# Mouseion

Mouseion is a local-first personal knowledge base. It runs as a single Python daemon, stores documents and chunks in SQLite with `sqlite-vec` and FTS5, embeds chunks through local Ollama, and exposes retrieval/admin operations through HTTP MCP tools plus a small web UI.

## Runtime

- Python 3.13
- uv
- SQLite via APSW, with `sqlite-vec` for vector search
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

## Configuration

Settings are loaded from `.env` and environment variables:

| Variable                       | Default                  |
| ------------------------------ | ------------------------ |
| `MOUSEION_HOST`                | `127.0.0.1`              |
| `MOUSEION_PORT`                | `7778`                   |
| `MOUSEION_DATA_DIR`            | `./data`                 |
| `MOUSEION_REPOS_DIR`           | `./repos`                |
| `OLLAMA_HOST`                  | `http://127.0.0.1:11434` |
| `EMBEDDING_MODEL`              | `nomic-embed-text`       |
| `CHUNK_TARGET_TOKENS`          | `512`                    |
| `CHUNK_MAX_TOKENS`             | `1024`                   |
| `CHUNK_MIN_TOKENS`             | `100`                    |
| `SIMILARITY_THRESHOLD`         | `0.82`                   |
| `SIMILARITY_TOP_K`             | `10`                     |
| `SIMILAR_EDGE_RECOMPUTE_HOURS` | `24`                     |
| `RRF_K`                        | `60`                     |
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
uv run mypy mouseion
uv run pytest
```
