"""Task-local PostgreSQL database lifecycle management."""

from __future__ import annotations

import hashlib
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import sql

from agentbrew.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PostgresConnection:
    """Administrative connection settings for the shared local PostgreSQL server."""

    host: str = "localhost"
    port: int = 5432
    username: str = "postgres"
    password: str = "password"
    admin_database: str = "postgres"
    docker_container: str = "mcpmark-postgres"

    def psycopg(self, *, database: str | None = None) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.username,
            "password": self.password,
            "database": database or self.admin_database,
        }

    def url(self, database: str) -> str:
        return (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{database}"
        )


@dataclass
class PostgresTaskState:
    """Resources created for one benchmark or trajectory task."""

    category: str
    task_id: str
    database_name: str
    database_url: str
    prepare_script: Path | None = None


class PostgresStateManager:
    """Clone one isolated logical database per task inside a shared Docker server.

    Template databases are immutable.  Tasks that use the same template are grouped
    by :class:`PostgresEnvironment`, while an advisory lock also protects cloning and
    first-time restore across independent AgentBrew processes.
    """

    DATABASE_PREFIX = "agentbrew_"

    def __init__(
        self,
        connection: PostgresConnection,
        *,
        backup_dir: str | Path,
        sql_dir: str | Path | None = None,
        prepare_root: str | Path | None = None,
        restore_timeout_seconds: int = 1800,
    ) -> None:
        self.connection = connection
        self.backup_dir = Path(backup_dir).expanduser()
        self.sql_dir = Path(sql_dir).expanduser() if sql_dir else None
        self.prepare_root = (
            Path(prepare_root).expanduser()
            if prepare_root
            else Path(__file__).parent / "benchmark" / "evaluator"
        )
        self.restore_timeout_seconds = restore_timeout_seconds
        self.created_databases: set[str] = set()
        self._test_connection()

    def _connect(self, database: str | None = None):
        return psycopg2.connect(**self.connection.psycopg(database=database))

    def _test_connection(self) -> None:
        conn = self._connect()
        conn.close()

    @staticmethod
    def _safe_identifier(value: str, *, fallback: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
        return cleaned or fallback

    def _task_database_name(self, category: str, task_id: str) -> str:
        category_slug = self._safe_identifier(category, fallback="postgres")[:20]
        task_slug = self._safe_identifier(task_id, fallback="task")[:16]
        suffix = uuid.uuid4().hex[:12]
        return f"{self.DATABASE_PREFIX}{category_slug}_{task_slug}_{suffix}"[:63]

    @staticmethod
    def _lock_key(template: str) -> int:
        raw = hashlib.blake2b(template.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(raw, byteorder="big", signed=True)

    def _database_exists(self, database: str, conn=None) -> bool:
        owns_connection = conn is None
        conn = conn or self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
                return cursor.fetchone() is not None
        finally:
            if owns_connection:
                conn.close()

    def _create_empty_database(self, database: str, conn=None) -> None:
        owns_connection = conn is None
        conn = conn or self._connect()
        conn.autocommit = True
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
                )
        finally:
            if owns_connection:
                conn.close()

    def _build_pg_command(self, tool: str, args: list[str]) -> list[str]:
        executable = shutil.which(tool)
        if executable:
            return [executable, *args]
        docker = shutil.which("docker")
        if not docker:
            raise FileNotFoundError(f"Neither {tool!r} nor Docker is available")
        return [
            docker,
            "exec",
            "-i",
            "-e",
            f"PGPASSWORD={self.connection.password}",
            self.connection.docker_container,
            tool,
            *args,
        ]

    def _run_pg_tool(
        self,
        tool: str,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        env = dict(os.environ)
        env["PGPASSWORD"] = self.connection.password
        result = subprocess.run(
            self._build_pg_command(tool, args),
            input=input_bytes,
            env=env,
            capture_output=True,
            timeout=self.restore_timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"{tool} failed: {stderr.strip()}")
        return result

    def _restore_template(self, category: str, conn) -> None:
        backup = self.backup_dir / f"{category}.backup"
        sql_seed = self.sql_dir / f"{category}.sql" if self.sql_dir else None
        compressed_sql_seed = (
            self.sql_dir / f"{category}.sql.gz" if self.sql_dir else None
        )
        self._create_empty_database(category, conn)
        common = [
            "-h", self.connection.host,
            "-p", str(self.connection.port),
            "-U", self.connection.username,
            "-d", category,
        ]
        try:
            if backup.exists():
                self._run_pg_tool(
                    "pg_restore",
                    [*common, "-v"],
                    input_bytes=backup.read_bytes(),
                )
            elif sql_seed and sql_seed.exists():
                self._restore_sql_to_database(category, sql_seed)
                self.cache_backup_from_database(category)
            elif compressed_sql_seed and compressed_sql_seed.exists():
                with gzip.open(compressed_sql_seed, "rb") as seed_file, tempfile.NamedTemporaryFile(
                    suffix=".sql", delete=False
                ) as expanded_file:
                    shutil.copyfileobj(seed_file, expanded_file)
                    expanded_path = Path(expanded_file.name)
                try:
                    self._restore_sql_to_database(category, expanded_path)
                    self.cache_backup_from_database(category)
                finally:
                    expanded_path.unlink(missing_ok=True)
            else:
                raise FileNotFoundError(
                    f"No PostgreSQL seed found for {category!r}; checked {backup}"
                    + (f", {sql_seed}, and {compressed_sql_seed}" if sql_seed else "")
                )
        except Exception:
            self._drop_database(category, allow_template=True)
            raise

    @staticmethod
    def _iter_sql_statements(sql_path: Path):
        """Yield statements using the generated seed files' line-ending format."""
        buffer: list[str] = []
        with sql_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                buffer.append(line)
                if line.rstrip().endswith(";"):
                    statement = "".join(buffer).strip()
                    buffer = []
                    if statement:
                        yield statement
        trailing = "".join(buffer).strip()
        if trailing:
            yield trailing

    @staticmethod
    def _normalize_alter_table_constraint(
        statement: str,
        constraint_counts: dict[tuple[str, str], int],
    ) -> str:
        pattern = re.compile(
            r'^(ALTER\s+TABLE\s+.+?\s+ADD\s+CONSTRAINT\s+)(?:"([^"]+)"|([^\s"]+)"?)(\s+FOREIGN\s+KEY.*)$',
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.match(statement.strip())
        if not match:
            return statement
        prefix, quoted_name, bare_name, suffix = match.groups()
        base_name = quoted_name or bare_name or "fk_constraint"
        table_match = re.match(
            r'^ALTER\s+TABLE\s+(".*?"|\S+)',
            statement.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        table_name = table_match.group(1) if table_match else "table"
        key = (table_name, base_name)
        constraint_counts[key] = constraint_counts.get(key, 0) + 1
        suffix_index = constraint_counts[key]
        constraint_name = base_name if suffix_index == 1 else f"{base_name}_{suffix_index}"
        return f'{prefix}"{constraint_name}"{suffix}'

    def _extract_inline_foreign_keys_from_create_table(
        self,
        statement: str,
        constraint_counts: dict[tuple[str, str], int],
    ) -> tuple[str, list[str]]:
        create_match = re.match(
            r'^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(".*?"|\S+)\s*\((.*)\)\s*;\s*$',
            statement.strip(),
            re.IGNORECASE | re.DOTALL,
        )
        if not create_match:
            return statement, []
        table_name = create_match.group(1)
        lines = statement.splitlines()
        if len(lines) < 3:
            return statement, []

        def build_alter(definition: str, index: int) -> str:
            definition = definition.rstrip(",").strip()
            constraint_match = re.match(
                r'^\s*CONSTRAINT\s+(".*?"|\S+)\s+(FOREIGN\s+KEY.*)$',
                definition,
                re.IGNORECASE | re.DOTALL,
            )
            if constraint_match:
                constraint_name, fk_clause = constraint_match.groups()
                raw = f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} {fk_clause};"
            else:
                generated_name = table_name.strip('"').replace(".", "_").replace('"."', "_")
                raw = (
                    f'ALTER TABLE {table_name} ADD CONSTRAINT "{generated_name}_fk_{index}" '
                    f"{definition};"
                )
            return self._normalize_alter_table_constraint(raw, constraint_counts)

        kept_lines: list[str] = []
        alters: list[str] = []
        for index, line in enumerate(lines[1:-1], start=1):
            stripped = line.strip()
            upper = stripped.upper()
            if "FOREIGN KEY" in upper and "REFERENCES" in upper and (
                upper.startswith("FOREIGN KEY") or upper.startswith("CONSTRAINT ")
            ):
                alters.append(build_alter(stripped, index))
            else:
                kept_lines.append(line.rstrip())
        if not alters:
            return statement, []
        normalized_body = []
        for index, line in enumerate(kept_lines):
            stripped = line.rstrip().rstrip(",")
            normalized_body.append(
                f"{stripped}," if index < len(kept_lines) - 1 else stripped
            )
        return "\n".join([lines[0].rstrip(), *normalized_body, lines[-1].rstrip()]), alters

    def _write_reordered_sql_files(self, sql_path: Path) -> tuple[Path, Path]:
        base_tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".sql", delete=False
        )
        alter_tmp = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".sql", delete=False
        )
        constraint_counts: dict[tuple[str, str], int] = {}
        try:
            for statement in self._iter_sql_statements(sql_path):
                normalized = statement.strip()
                if not normalized:
                    continue
                normalized += ";" if not normalized.endswith(";") else ""
                if re.match(r"^\s*ALTER\s+TABLE\b", normalized, re.IGNORECASE):
                    alter_tmp.write(
                        self._normalize_alter_table_constraint(
                            normalized, constraint_counts
                        )
                        + "\n"
                    )
                else:
                    normalized, extracted = self._extract_inline_foreign_keys_from_create_table(
                        normalized, constraint_counts
                    )
                    base_tmp.write(normalized + "\n")
                    for alter_statement in extracted:
                        alter_tmp.write(alter_statement + "\n")
        finally:
            base_tmp.close()
            alter_tmp.close()
        return Path(base_tmp.name), Path(alter_tmp.name)

    def _restore_sql_to_database(self, database: str, sql_file: Path) -> None:
        base_sql, alter_sql = self._write_reordered_sql_files(sql_file)
        common = [
            "-h", self.connection.host,
            "-p", str(self.connection.port),
            "-U", self.connection.username,
            "-d", database,
            "-v", "ON_ERROR_STOP=1",
        ]
        try:
            self._run_pg_tool("psql", common, input_bytes=base_sql.read_bytes())
            self._apply_alter_statements(database, alter_sql)
            self._sync_serial_sequences(database)
        finally:
            base_sql.unlink(missing_ok=True)
            alter_sql.unlink(missing_ok=True)

    def _apply_alter_statements(self, database: str, alter_sql: Path) -> None:
        if alter_sql.stat().st_size == 0:
            return
        conn = self._connect(database)
        conn.autocommit = False
        skipped_count = 0
        try:
            with conn.cursor() as cursor:
                for statement in self._iter_sql_statements(alter_sql):
                    try:
                        cursor.execute(statement)
                        conn.commit()
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        conn.rollback()
                        skipped_count += 1
                        logger.warning(
                            "Skipping invalid ALTER statement for %s: %s | %s",
                            database,
                            statement,
                            exc,
                        )
        finally:
            conn.close()
        if skipped_count:
            logger.warning(
                "Skipped %s invalid ALTER statements while importing %s",
                skipped_count,
                database,
            )

    def _sync_serial_sequences(self, database: str) -> None:
        conn = self._connect(database)
        conn.autocommit = True
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT table_schema, table_name, column_name
                    FROM information_schema.columns
                    WHERE column_default LIKE 'nextval(%'
                    """
                )
                for schema_name, table_name, column_name in cursor.fetchall():
                    relation_text = f'"{schema_name}"."{table_name}"'
                    cursor.execute(
                        "SELECT pg_get_serial_sequence(%s, %s)",
                        (relation_text, column_name),
                    )
                    sequence_name = cursor.fetchone()[0]
                    if not sequence_name:
                        continue
                    cursor.execute(
                        sql.SQL(
                            """
                            SELECT setval(
                                %s,
                                GREATEST(COALESCE(MAX({column}), 1), 1),
                                GREATEST(COALESCE(MAX({column}), 1), 1) > 1
                            )
                            FROM {schema}.{table}
                            """
                        ).format(
                            column=sql.Identifier(column_name),
                            schema=sql.Identifier(schema_name),
                            table=sql.Identifier(table_name),
                        ),
                        (sequence_name,),
                    )
        finally:
            conn.close()

    def _ensure_template(self, category: str, conn) -> bool:
        if self._database_exists(category, conn):
            return True
        backup = self.backup_dir / f"{category}.backup"
        sql_seed = self.sql_dir / f"{category}.sql" if self.sql_dir else None
        compressed_sql_seed = (
            self.sql_dir / f"{category}.sql.gz" if self.sql_dir else None
        )
        if (
            not backup.exists()
            and not (sql_seed and sql_seed.exists())
            and not (compressed_sql_seed and compressed_sql_seed.exists())
        ):
            return False
        self._restore_template(category, conn)
        return True

    def ensure_template_database(self, category: str) -> bool:
        """Ensure a seed database exists, matching the migrated conversion flow."""
        category = self._safe_identifier(category, fallback="postgres")
        conn = self._connect()
        conn.autocommit = True
        lock_key = self._lock_key(category)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
                try:
                    return self._ensure_template(category, conn)
                finally:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
        finally:
            conn.close()

    def cache_backup_from_database(self, database: str) -> Path:
        """Dump an existing database into a reusable custom-format backup."""
        backup = self.backup_dir / f"{database}.backup"
        if backup.exists():
            return backup
        backup.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "-h", self.connection.host,
            "-p", str(self.connection.port),
            "-U", self.connection.username,
            "-d", database,
            "-Fc",
        ]
        if shutil.which("pg_dump"):
            self._run_pg_tool("pg_dump", [*args, "-f", str(backup)])
        else:
            result = self._run_pg_tool("pg_dump", args)
            backup.write_bytes(result.stdout)
        logger.info("Cached local PostgreSQL backup %s", backup)
        return backup

    def _prepare_script(self, category: str, task_id: str) -> Path | None:
        path = (
            self.prepare_root
            / category
            / task_id
            / "prepare_environment.py"
        )
        return path if path.exists() else None

    def _run_prepare_script(self, script: Path, database: str) -> None:
        env = dict(os.environ)
        env.update(
            {
                "POSTGRES_HOST": self.connection.host,
                "POSTGRES_PORT": str(self.connection.port),
                "POSTGRES_USERNAME": self.connection.username,
                "POSTGRES_PASSWORD": self.connection.password,
                "POSTGRES_DATABASE": database,
            }
        )
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(script.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.restore_timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PostgreSQL prepare script failed ({script}): {result.stderr.strip()}"
            )

    def create_task_state(self, category: str, task_id: str) -> PostgresTaskState:
        """Create a database cloned from the category template."""
        category = self._safe_identifier(category, fallback="postgres")
        database = self._task_database_name(category, task_id)
        prepare_script = self._prepare_script(category, task_id)
        conn = self._connect()
        conn.autocommit = True
        lock_key = self._lock_key(category)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
                template_exists = self._ensure_template(category, conn)
                if template_exists:
                    cursor.execute(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE datname = %s AND pid <> pg_backend_pid()",
                        (category,),
                    )
                    cursor.execute(
                        sql.SQL("CREATE DATABASE {} WITH TEMPLATE {}").format(
                            sql.Identifier(database), sql.Identifier(category)
                        )
                    )
                else:
                    self._create_empty_database(database, conn)
                cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
        except Exception:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
            except Exception:  # pragma: no cover - best effort unlock on broken connection
                pass
            raise
        finally:
            conn.close()

        self.created_databases.add(database)
        try:
            if prepare_script:
                self._run_prepare_script(prepare_script, database)
        except Exception:
            self.cleanup_database(database)
            raise

        return PostgresTaskState(
            category=category,
            task_id=task_id,
            database_name=database,
            database_url=self.connection.url(database),
            prepare_script=prepare_script,
        )

    def _drop_database(self, database: str, *, allow_template: bool = False) -> None:
        if not allow_template and not database.startswith(self.DATABASE_PREFIX):
            raise ValueError(f"Refusing to drop non-AgentBrew database: {database}")
        conn = self._connect()
        conn.autocommit = True
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND pid <> pg_backend_pid()",
                    (database,),
                )
                cursor.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database))
                )
        finally:
            conn.close()

    def cleanup_database(self, database: str) -> None:
        self._drop_database(database)
        self.created_databases.discard(database)
        logger.info("Dropped isolated PostgreSQL task database %s", database)
