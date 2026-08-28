"""Utility functions for Notion API interactions in verification scripts."""
# pylint: disable=duplicate-code,import-error,astroid-error
import os
import sys
from typing import List, Tuple
from dotenv import load_dotenv
from notion_client import Client


class SubCriteriaCollector:
    """
    Collects sub-criterion pass/fail results so that ALL checks run
    (no early exit on first failure) and prints a summary at the end.

    Usage::
        sc = SubCriteriaCollector("Task Name")
        if not sc.check("Database found", db_id is not None, "DB not found"):
            sc.fail_remaining(["Schema valid", "Has 3 entries"])
            return sc.summary()
        sc.check("Schema valid", _check_schema(...), "Bad schema")
        sc.check("Has 3 entries", len(rows) >= 3, "Only N entries")
        return sc.summary()
    """

    def __init__(self, task_name: str = ""):
        self.task_name = task_name
        self._items: List[Tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, error_msg: str = "") -> bool:
        """Record one sub-criterion.  Returns *condition* so callers can branch."""
        ok = bool(condition)
        self._items.append((name, ok, "" if ok else error_msg))
        return ok

    def fail_remaining(self, names: List[str], reason: str = "prerequisite failed") -> None:
        """Mark a list of checks as failed because a prerequisite check failed."""
        for name in names:
            self._items.append((name, False, f"[skipped – {reason}]"))

    def summary(self) -> Tuple[bool, str]:
        """Print a per-task summary and return *(all_passed, structured_reason_string)*.

        The reason string encodes sub-criteria details so that report generators can
        display per-check pass rates and failure messages.  Format::

            [SUBCRITERIA:n_passed/total]
            PASS | Check name 1
            FAIL | Check name 2 | error message
        """
        total = len(self._items)
        n_passed = sum(1 for _, ok, _ in self._items if ok)
        all_passed = n_passed == total

        print(f"\n{'─' * 62}", file=sys.stderr, flush=True)
        label = f"  Task: {self.task_name}" if self.task_name else "  Task result"
        status = "PASS" if all_passed else "FAIL"
        print(f"{label}  [{status}]", file=sys.stderr, flush=True)
        print(f"  Sub-criteria: {n_passed}/{total} passed", file=sys.stderr, flush=True)
        for name, ok, msg in self._items:
            mark = "✓" if ok else "✗"
            suffix = f"  ({msg})" if (not ok and msg) else ""
            print(f"    {mark} {name}{suffix}", file=sys.stderr, flush=True)
        print(f"{'─' * 62}", file=sys.stderr, flush=True)

        # Build a structured reason string parseable by the report generator.
        lines = [f"[SUBCRITERIA:{n_passed}/{total}]"]
        for name, ok, msg in self._items:
            if ok:
                lines.append(f"PASS | {name}")
            elif msg:
                lines.append(f"FAIL | {name} | {msg}")
            else:
                lines.append(f"FAIL | {name}")
        return all_passed, "\n".join(lines)


def get_notion_client():
    """Get a Notion API client instance using credentials from environment."""
    # Construct the absolute path to the .env file in the project root
    load_dotenv(dotenv_path=".mcp_env")
    api_key = os.getenv("EVAL_NOTION_API_KEY")
    if not api_key:
        print(
            "Error: EVAL_NOTION_API_KEY not found in environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)
    return Client(auth=api_key)


def _find_object(notion: Client, title: str, object_type: str):
    """Generic helper to find a Notion page or database by title.

    Args:
        notion: Authenticated Notion Client.
        title: Title (or partial title) to search for.
        object_type: Either "page" or "database".

    Returns:
        The ID string if found, otherwise None.
    """
    search_type = "data_source" if object_type == "database" else object_type
    search_results = (
        notion.search(
            query=title, filter={"property": "object", "value": search_type}
        ).get("results")
        or []
    )

    if not search_results:
        return None

    # Shortcut when there is only one match
    if len(search_results) == 1:
        return search_results[0]["id"]

    # Attempt to find a case-insensitive match on the title field
    for result in search_results:
        if object_type == "page":
            # Pages store their title inside the "properties.title.title" rich text list
            title_rich_texts = (
                result.get("properties", {}).get("title", {}).get("title", [])
            )
        else:  # database / data_source
            title_rich_texts = result.get("title", [])

        for text_obj in title_rich_texts:
            if title.lower() in text_obj.get("plain_text", "").lower():
                return result["id"]

    # Fallback: return the first result
    return search_results[0]["id"]


def find_page(notion: Client, page_title: str):
    """Finds a page by title. Wrapper around _find_object with object_type='page'."""
    return _find_object(notion, page_title, "page")


def get_page_by_id(notion: Client, page_id: str):
    """Gets a page by its ID. Returns the page object if found, None otherwise."""
    try:
        return notion.pages.retrieve(page_id=page_id)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def find_page_by_id(notion: Client, page_id: str):
    """Finds a page by its ID and returns the ID if it exists, None otherwise."""
    try:
        notion.pages.retrieve(page_id=page_id)
        return page_id
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def find_database_by_id(notion: Client, database_id: str):
    """Finds a database by its ID and returns the ID if it exists, None otherwise."""
    try:
        retrieve_database(notion, database_id)
        return database_id
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def find_page_or_database_by_id(notion: Client, object_id: str):
    """
    Finds either a page or database by ID. Returns a tuple (object_id, object_type)
    where object_type is either 'page' or 'database', or (None, None) if not found.
    """
    # Try as page first
    try:
        notion.pages.retrieve(page_id=object_id)
        return (object_id, "page")
    except (ValueError, KeyError, TypeError, AttributeError):
        pass

    # Try as database
    try:
        retrieve_database(notion, object_id)
        return (object_id, "database")
    except (ValueError, KeyError, TypeError, AttributeError):
        pass

    return (None, None)


def find_database(notion: Client, db_title: str):
    """Finds a database by title. Wrapper around _find_object with object_type='database'."""
    return _find_object(notion, db_title, "database")


def retrieve_database(notion: Client, database_id: str):
    """
    Retrieve a Notion database-like object, accepting either a modern
    `data_source_id` or a legacy `database_id`.
    """
    try:
        return notion.data_sources.retrieve(data_source_id=database_id)
    except Exception:  # pylint: disable=broad-exception-caught
        database = notion.databases.retrieve(database_id=database_id)

    if database.get("properties"):
        return database
    for data_source in database.get("data_sources", []):
        data_source_id = data_source.get("id")
        if data_source_id:
            return notion.data_sources.retrieve(data_source_id=data_source_id)
    return database


def query_database(notion: Client, database_id: str, **kwargs):
    """
    Query a Notion database-like object, accepting either a modern
    `data_source_id` or a legacy `database_id`.
    """
    query_kwargs = dict(kwargs)
    if query_kwargs.get("archived") is False:
        query_kwargs.pop("archived")

    try:
        return notion.data_sources.query(
            data_source_id=database_id, **query_kwargs
        )
    except Exception as data_source_error:  # pylint: disable=broad-exception-caught
        try:
            database = notion.databases.retrieve(database_id=database_id)
            for data_source in database.get("data_sources", []):
                data_source_id = data_source.get("id")
                if data_source_id:
                    return notion.data_sources.query(
                        data_source_id=data_source_id, **query_kwargs
                    )
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        try:
            return notion.request(
                path=f"databases/{database_id}/query",
                method="POST",
                body=query_kwargs or None,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            raise data_source_error


def resolve_database_id_with_fallback(
    notion: Client, database_id: str | None = None, db_title: str | None = None
):
    """
    Resolve a queryable database/data source ID.

    Prefer the provided ID to preserve existing verifier behavior. If it cannot be
    retrieved as a database/data source, fall back to finding a database by title.
    Returns None when neither strategy succeeds.
    """
    if database_id:
        try:
            retrieve_database(notion, database_id)
            return database_id
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    if db_title:
        try:
            resolved = find_database(notion, db_title)
            if resolved:
                retrieve_database(notion, resolved)
                return resolved
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    return None


def query_database_with_fallback(
    notion: Client, database_id: str | None = None, db_title: str | None = None, **kwargs
):
    """
    Query a database while preserving old verifier behavior first and falling back
    to a title-based lookup when the provided ID is not queryable.
    """
    first_error = None
    if database_id:
        try:
            return query_database(notion, database_id, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            first_error = exc

    # If the original id is retrievable but not actually queryable (common for
    # child_database block ids), prefer resolving a fresh database id by title.
    title_resolved_id = None
    if db_title:
        try:
            candidate_id = find_database(notion, db_title)
            if candidate_id and candidate_id != database_id:
                retrieve_database(notion, candidate_id)
                title_resolved_id = candidate_id
        except Exception:  # pylint: disable=broad-exception-caught
            title_resolved_id = None

    if title_resolved_id:
        return query_database(notion, title_resolved_id, **kwargs)

    resolved_id = resolve_database_id_with_fallback(
        notion, database_id=database_id, db_title=db_title
    )
    if resolved_id and resolved_id != database_id:
        return query_database(notion, resolved_id, **kwargs)

    if first_error is not None:
        raise first_error
    raise ValueError(
        f"Unable to resolve a queryable database id for title={db_title!r}, id={database_id!r}"
    )


def find_database_in_block(notion: Client, block_id: str, db_title: str):
    """
    Recursively find a database by title within a block.
    """
    blocks = notion.blocks.children.list(block_id=block_id).get("results")
    for block in blocks:
        if (
            block.get("type") == "child_database"
            and block.get("child_database", {}).get("title") == db_title
        ):
            return find_database(notion, db_title) or block["id"]
        if block.get("has_children"):
            db_id = find_database_in_block(notion, block["id"], db_title)
            if db_id:
                return db_id
    return None


def get_all_blocks_recursively(notion: Client, block_id: str):
    """
    Recursively fetches all blocks from a starting block ID and its children,
    returning a single flat list of block objects.
    """
    all_blocks = []
    try:
        direct_children = notion.blocks.children.list(block_id=block_id).get(
            "results", []
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(
            f"[verify warn] skipping unreadable block {block_id}: {exc}",
            file=sys.stderr,
        )
        return []

    for block in direct_children:
        all_blocks.append(block)
        if block.get("has_children"):
            try:
                all_blocks.extend(get_all_blocks_recursively(notion, block["id"]))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(
                    f"[verify warn] skipping child subtree {block.get('id')}: {exc}",
                    file=sys.stderr,
                )

    return all_blocks


def get_block_plain_text(block):
    """
    Safely extract plain_text from a block (paragraph, heading, etc.).
    """
    block_type = block.get("type")
    if not block_type:
        return ""

    block_content = block.get(block_type)
    if not block_content:
        return ""

    rich_text_list = block_content.get("rich_text", [])
    plain_text = "".join([rt.get("plain_text", "") for rt in rich_text_list])

    return plain_text
