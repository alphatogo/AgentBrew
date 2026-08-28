"""Verification module for Employee Onboarding task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from typing import Dict, Optional, Set
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def _check_db_schema(db_props: Dict[str, Dict], required: Dict[str, str]) -> bool:
    """Return True if every required property exists with the correct type."""
    for prop_name, expected_type in required.items():
        if prop_name not in db_props:
            print(
                f"Error: Property '{prop_name}' missing from database.", file=sys.stderr
            )
            return False
        actual_type = db_props[prop_name]["type"]
        if actual_type != expected_type:
            msg = (f"Error: Property '{prop_name}' has type "
                   f"'{actual_type}', expected '{expected_type}'.")
            print(msg, file=sys.stderr)
            return False
    return True


def verify(notion: Client, _main_id: Optional[str] = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """Programmatically verify the onboarding system described in description.md."""

    sc = SubCriteriaCollector("Employee Onboarding")

    # Constants for verification
    DB_TITLE = "Employee Onboarding Checklist"  # pylint: disable=invalid-name
    HUB_PAGE_TITLE = "Onboarding Hub"  # pylint: disable=invalid-name
    DEPARTMENT_OPTIONS: Set[str] = {  # pylint: disable=invalid-name
        "Product",
        "Marketing",
        "Sales",
        "HR",
        "Engineering",
    }
    REQUIRED_DB_PROPERTIES = {  # pylint: disable=invalid-name
        "Employee Name": "title",
        "Start Date": "date",
        "Department": "select",
    }

    # 1. Locate onboarding database
    db_id = notion_utils.find_database(notion, DB_TITLE)
    if not sc.check("Database exists", bool(db_id), f"Database '{DB_TITLE}' not found"):
        sc.fail_remaining([
            "Database retrievable", "Schema valid", "Dept options correct",
            "At least 3 entries"
        ], "database not found")
    else:
        # 2. Retrieve database
        db_obj = None
        db_retrieve_ok = False
        db_retrieve_err = ""
        try:
            db_obj = notion_utils.retrieve_database(notion, db_id)
            db_retrieve_ok = True
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            db_retrieve_err = str(exc)

        if not sc.check("Database retrievable", db_retrieve_ok,
                        f"Error retrieving database: {db_retrieve_err}"):
            sc.fail_remaining(["Schema valid", "Dept options correct", "At least 3 entries"],
                              "db not retrievable")
        else:
            db_props = db_obj.get("properties", {})
            # 3. Schema
            sc.check("Schema valid", _check_db_schema(db_props, REQUIRED_DB_PROPERTIES),
                     "Database schema validation failed")

            # 4. Dept options
            dept_options = {opt["name"] for opt in db_props["Department"]["select"]["options"]}
            if not sc.check("Dept options correct",
                            DEPARTMENT_OPTIONS.issubset(dept_options),
                            f"Missing dept options: {sorted(DEPARTMENT_OPTIONS - dept_options)}"):
                pass  # non-blocking

            # 5. At least 3 entries
            db_pages = []
            db_query_ok = False
            db_query_err = ""
            try:
                db_pages = notion_utils.query_database(notion, db_id).get("results", [])
                db_query_ok = True
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                db_query_err = str(exc)

            if not db_query_ok:
                sc.check("At least 3 entries", False, f"Error querying database: {db_query_err}")
            else:
                sc.check("At least 3 entries", len(db_pages) >= 3,
                         f"Only {len(db_pages)} entries found")

    # 6. Locate Onboarding Hub page (independent of db checks)
    hub_page_id = notion_utils.find_page(notion, HUB_PAGE_TITLE)
    if not sc.check("Onboarding Hub page exists", bool(hub_page_id),
                    f"Page '{HUB_PAGE_TITLE}' not found"):
        sc.fail_remaining([
            "DB embedded in Onboarding Hub", "At least 3 link mentions",
            "At least 7 numbered list steps", "At least 3 todo items"
        ], "hub page not found")
        return sc.summary()

    # 7. Ensure the onboarding database is embedded in the hub page
    embedded_db_id = notion_utils.find_database_in_block(
        notion, hub_page_id, DB_TITLE
    )
    sc.check(
        "DB embedded in Onboarding Hub",
        bool(db_id) and bool(embedded_db_id) and embedded_db_id == db_id,
        "The Employee Onboarding Checklist database is not embedded in the Onboarding Hub page",
    )

    # 8. Analyse blocks within the hub page
    all_blocks = notion_utils.get_all_blocks_recursively(notion, hub_page_id)

    seen_link_targets: Set[str] = set()
    numbered_list_count = 0
    todo_count = 0

    for blk in all_blocks:  # pylint: disable=too-many-nested-blocks
        blk_type = blk.get("type")

        # Direct link-to-page blocks
        if blk_type == "link_to_page":
            info = blk.get("link_to_page", {})
            target_id = info.get("page_id") or info.get("database_id")
            if target_id:
                seen_link_targets.add(target_id)
            continue

        # Rich-text mentions inside content blocks
        if blk_type in {
            "paragraph",
            "numbered_list_item",
            "bulleted_list_item",
            "to_do",
        }:
            content = blk.get(blk_type, {})
            for rt in content.get("rich_text", []):
                if rt.get("type") == "mention":
                    mention = rt.get("mention", {})
                    if mention.get("type") in {"page", "database"}:
                        target_id = mention.get("page", {}).get("id") or mention.get(
                            "database", {}
                        ).get("id")
                        if target_id:
                            seen_link_targets.add(target_id)

        # Count numbered list items
        if blk_type == "numbered_list_item":
            numbered_list_count += 1

        # Count to-do items in Feedback Form
        if blk_type == "to_do":
            todo_count += 1

    sc.check("At least 3 link mentions", len(seen_link_targets) >= 3,
             f"Fewer than 3 linked mentions found (found {len(seen_link_targets)})")
    sc.check("At least 7 numbered list steps", numbered_list_count >= 7,
             f"Numbered list contains only {numbered_list_count} steps")
    sc.check("At least 3 todo items", todo_count >= 3,
             f"Feedback Form section contains only {todo_count} to-do items")

    return sc.summary()


def main():
    """Main verification function."""
    notion = notion_utils.get_notion_client()
    main_id = sys.argv[1] if len(sys.argv) > 1 else None
    success, _error_msg = verify(notion, main_id)
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
