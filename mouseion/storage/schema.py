from __future__ import annotations

SCHEMA_VERSION = "3"

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS documents(
      id TEXT PRIMARY KEY,
      type TEXT CHECK(type IN('document','memory','url','file')),
      title TEXT,
      source TEXT,
      content_hash TEXT,
      created_at TEXT,
      updated_at TEXT,
      metadata TEXT DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_doc_source ON documents(source)",
    "CREATE INDEX IF NOT EXISTS idx_doc_title_nocase ON documents(title COLLATE NOCASE)",
    "CREATE INDEX IF NOT EXISTS idx_doc_type_source ON documents(type, source)",
    "CREATE INDEX IF NOT EXISTS idx_doc_type_content_hash ON documents(type, content_hash)",
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_unique_type_source
    ON documents(type, source)
    WHERE type != 'memory'
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_unique_memory_content_hash
    ON documents(type, content_hash)
    WHERE type = 'memory'
    """,
    """
    CREATE TABLE IF NOT EXISTS tags(
      document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
      tag TEXT,
      PRIMARY KEY(document_id, tag)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks(
      id INTEGER PRIMARY KEY,
      document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
      content TEXT,
      chunk_index INTEGER,
      token_count INTEGER,
      created_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id)",
    """
    CREATE TABLE IF NOT EXISTS chunk_vectors(
      chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
      embedding BLOB NOT NULL
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
      content,
      content='chunks',
      content_rowid='id',
      tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, content) VALUES(new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, content)
      VALUES('delete', old.id, old.content);
      DELETE FROM chunk_vectors WHERE chunk_id = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, content)
      VALUES('delete', old.id, old.content);
      INSERT INTO chunks_fts(rowid, content) VALUES(new.id, new.content);
    END
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
      title,
      source,
      tags,
      metadata,
      content,
      tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_search_ad AFTER DELETE ON chunks BEGIN
      DELETE FROM search_fts WHERE rowid = old.id;
    END
    """,
    """
    CREATE TABLE IF NOT EXISTS stats_counters(
      name TEXT PRIMARY KEY,
      value INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stats_document_types(
      type TEXT PRIMARY KEY,
      value INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_stats_ai AFTER INSERT ON documents BEGIN
      INSERT INTO stats_counters(name, value) VALUES('documents', 1)
      ON CONFLICT(name) DO UPDATE SET value = value + 1;
      INSERT INTO stats_document_types(type, value) VALUES(new.type, 1)
      ON CONFLICT(type) DO UPDATE SET value = value + 1;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_stats_ad AFTER DELETE ON documents BEGIN
      UPDATE stats_counters SET value = max(value - 1, 0) WHERE name = 'documents';
      UPDATE stats_document_types SET value = max(value - 1, 0) WHERE type = old.type;
      DELETE FROM stats_document_types WHERE value = 0;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_stats_au AFTER UPDATE OF type ON documents
    WHEN old.type != new.type
    BEGIN
      UPDATE stats_document_types SET value = max(value - 1, 0) WHERE type = old.type;
      DELETE FROM stats_document_types WHERE value = 0;
      INSERT INTO stats_document_types(type, value) VALUES(new.type, 1)
      ON CONFLICT(type) DO UPDATE SET value = value + 1;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_stats_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO stats_counters(name, value) VALUES('chunks', 1)
      ON CONFLICT(name) DO UPDATE SET value = value + 1;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_stats_ad AFTER DELETE ON chunks BEGIN
      UPDATE stats_counters SET value = max(value - 1, 0) WHERE name = 'chunks';
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tags_stats_ai AFTER INSERT ON tags BEGIN
      INSERT INTO stats_counters(name, value) VALUES('tags', 1)
      ON CONFLICT(name) DO UPDATE SET value = value + 1;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS tags_stats_ad AFTER DELETE ON tags BEGIN
      UPDATE stats_counters SET value = max(value - 1, 0) WHERE name = 'tags';
    END
    """,
    "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)",
]
