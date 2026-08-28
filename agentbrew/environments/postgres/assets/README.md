# PostgreSQL environment assets

This directory contains the database metadata and seed data needed by the
AgentBrew PostgreSQL environment. It is intentionally self-contained and does
not read from any adjacent source checkout at runtime.

- `metadata/`: schema summaries used for task sampling (162 databases).
- `postgres_state/`: PostgreSQL custom-format backups (140 databases).
- `postgres_sql/`: gzip-compressed SQL fallback seeds for the remaining 22
  databases.

`PostgresStateManager` prefers a `.backup`, then falls back to `.sql` or
`.sql.gz`. The SQL files are compressed to avoid committing hundreds of
megabytes of duplicate uncompressed dumps.
