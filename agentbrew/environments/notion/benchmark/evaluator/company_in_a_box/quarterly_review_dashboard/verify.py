"""Verification module for Quarterly Review Dashboard task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from typing import List
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def _contains_keywords(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return all(kw.lower() in lowered for kw in keywords)


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """Programmatically verify that the dashboard page and its contents meet the
    requirements described in description.md.
    """
    sc = SubCriteriaCollector("Quarterly Review Dashboard")

    DASHBOARD_TITLE = "Q4 2024 Business Review Dashboard"  # pylint: disable=invalid-name
    PARENT_PAGE_TITLE = "Company In A Box"  # pylint: disable=invalid-name
    CALL_OUT_KEYWORDS = ["latam", "enterprise", "employee engagement"]  # pylint: disable=invalid-name
    HEADING_DEPARTMENTS = [  # pylint: disable=invalid-name
        "Product", "Marketing", "Sales", "Human Resources"
    ]
    ACTION_ITEM_DEPARTMENTS = {  # pylint: disable=invalid-name
        "Product", "Marketing", "Sales", "HR"
    }
    REQUIRED_DB_PROPERTIES = {  # pylint: disable=invalid-name
        "Task Name": "title",
        "Department": "select",
        "Priority": "select",
        "Status": "status",
    }
    PRIORITY_OPTIONS = {"High", "Medium", "Low"}  # pylint: disable=invalid-name

    # 1. Locate the dashboard page
    page_id = None
    if main_id:
        found_id, obj_type = notion_utils.find_page_or_database_by_id(notion, main_id)
        if found_id and obj_type == "page":
            page_id = found_id

    if not page_id:
        page_id = notion_utils.find_page(notion, DASHBOARD_TITLE)

    if not sc.check("Dashboard page found", bool(page_id),
                    f"Page '{DASHBOARD_TITLE}' not found"):
        sc.fail_remaining([
            "Dashboard is child of Company In A Box", "Callout with keywords found",
            "All department headings found", "Action Items database found",
            "Database retrievable", "Database schema valid",
            "At least 5 action items", "All action items valid"
        ], "dashboard page not found")
        return sc.summary()

    # Optional: ensure it is a child of Company In A Box
    try:
        page_obj = notion.pages.retrieve(page_id=page_id)
        parent_id = page_obj.get("parent", {}).get("page_id")
        if parent_id:
            parent_page = notion.pages.retrieve(page_id=parent_id)
            parent_title_rt = (
                parent_page.get("properties", {}).get("title", {}).get("title", [])
            )
            parent_title = (
                parent_title_rt[0].get("plain_text") if parent_title_rt else None
            )
            sc.check("Dashboard is child of Company In A Box",
                     parent_title == PARENT_PAGE_TITLE,
                     f"Dashboard page is not a direct child of '{PARENT_PAGE_TITLE}'")
        else:
            sc.check("Dashboard is child of Company In A Box", True)  # can't determine
    except (ValueError, KeyError, TypeError, AttributeError):
        sc.check("Dashboard is child of Company In A Box", True)  # best-effort only

    # 2. Verify callout with keywords
    all_blocks = notion_utils.get_all_blocks_recursively(notion, page_id)
    callout_ok = False
    for block in all_blocks:
        if block.get("type") == "callout":
            callout_text = notion_utils.get_block_plain_text(block)
            if _contains_keywords(callout_text, CALL_OUT_KEYWORDS):
                callout_ok = True
                break
    sc.check("Callout with keywords found", callout_ok,
             "No callout found with all three Current Goal keywords (LATAM, Enterprise, Employee engagement)")

    # 3. Verify department section headings
    found_depts = set()
    for block in all_blocks:
        if block.get("type") in {"heading_1", "heading_2", "heading_3"}:
            heading_text = notion_utils.get_block_plain_text(block)
            for dept in HEADING_DEPARTMENTS:
                if dept.lower() in heading_text.lower():
                    found_depts.add(dept)
    missing = set(HEADING_DEPARTMENTS) - found_depts
    sc.check("All department headings found", not missing,
             f"Missing department headings: {', '.join(missing)}")

    # 4. Verify Action Items database exists and has correct schema
    db_id = notion_utils.find_database_in_block(notion, page_id, "Action Items")
    if not sc.check("Action Items database found", bool(db_id),
                    "Database 'Action Items' not found on the dashboard"):
        sc.fail_remaining([
            "Database retrievable", "Database schema valid",
            "At least 5 action items", "All action items valid"
        ], "Action Items database not found")
        return sc.summary()

    db = None
    db_retrieve_ok = False
    db_retrieve_err = ""
    try:
        db = notion_utils.retrieve_database(notion, db_id)
        db_retrieve_ok = True
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        db_retrieve_err = str(exc)

    if not sc.check("Database retrievable", db_retrieve_ok,
                    f"Unable to retrieve database: {db_retrieve_err}"):
        sc.fail_remaining([
            "Database schema valid", "At least 5 action items", "All action items valid"
        ], "database not retrievable")
        return sc.summary()

    db_props = db.get("properties", {})
    schema_ok = True
    schema_err = ""
    for prop_name, expected_type in REQUIRED_DB_PROPERTIES.items():
        if prop_name not in db_props:
            schema_ok = False
            schema_err = f"Property '{prop_name}' missing from database"
            break
        actual_type = db_props[prop_name]["type"]
        if isinstance(expected_type, list):
            if actual_type not in expected_type:
                schema_ok = False
                schema_err = (f"Property '{prop_name}' has type '{actual_type}', "
                              f"expected one of {expected_type}")
                break
        else:
            if actual_type != expected_type:
                schema_ok = False
                schema_err = (f"Property '{prop_name}' has type '{actual_type}', "
                              f"expected '{expected_type}'")
                break
        # Extra check for Priority options
        if prop_name == "Priority":
            options = {opt["name"] for opt in db_props[prop_name]["select"]["options"]}
            if not PRIORITY_OPTIONS.issubset(options):
                schema_ok = False
                schema_err = (f"Priority property options must include High/Medium/Low. "
                              f"Current options: {options}")
                break

    sc.check("Database schema valid", schema_ok, schema_err)

    # 5. Verify at least 5 action items exist
    pages = []
    pages_ok = False
    pages_err = ""
    try:
        pages = notion_utils.query_database(notion, db_id).get("results", [])
        pages_ok = True
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        pages_err = str(exc)

    if not pages_ok:
        sc.check("At least 5 action items", False, f"Error querying database pages: {pages_err}")
        sc.fail_remaining(["All action items valid"], "could not query database")
        return sc.summary()

    sc.check("At least 5 action items", len(pages) >= 5,
             f"Database contains only {len(pages)} action items")

    # 6. Validate each action item
    items_valid = True
    items_err = ""
    for page in pages:
        props = page.get("properties", {})

        # Task Name must be non-empty
        title_rt = props.get("Task Name", {}).get("title", [])
        task_name = title_rt[0].get("plain_text") if title_rt else ""
        if not task_name.strip():
            print(f"Error: Action item '{page.get('id')}' is missing a Task Name.",
                  file=sys.stderr)
            items_valid = False
            items_err = f"Action item '{page.get('id')}' is missing a Task Name"
            continue

        # Department must be valid
        dept_select = props.get("Department", {}).get("select", {}).get("name")
        if not dept_select or dept_select not in ACTION_ITEM_DEPARTMENTS:
            print(f"Error: Action item '{page.get('id')}' has invalid or missing Department value.",
                  file=sys.stderr)
            items_valid = False
            items_err = f"Action item '{page.get('id')}' has invalid or missing Department value"
            continue

        # Priority and Status must be set (any value)
        priority_val = props.get("Priority", {}).get("select", {}).get("name")
        status_val = props.get("Status", {}).get("status", {}).get("name")
        if not priority_val or not status_val:
            print(f"Error: Action item '{page.get('id')}' must have both Priority and Status set.",
                  file=sys.stderr)
            items_valid = False
            items_err = f"Action item '{page.get('id')}' must have both Priority and Status set"

    sc.check("All action items valid", items_valid, items_err)

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
