"""Verification module for Security Audit Ticket task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
import re
from typing import Optional

from notion_client import Client

from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def _get_title_text(page_properties: dict) -> str:
    """Extract the plain text of the first title property from a page."""
    for prop in page_properties.values():
        if prop.get("type") == "title":
            title_rich = prop.get("title", [])
            if title_rich:
                return title_rich[0].get("plain_text")
    return ""


def verify(notion: Client, main_id: Optional[str] = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """Verify that the automation created the expected security audit ticket."""

    sc = SubCriteriaCollector("Security Audit Ticket")

    # ----------------------------------------------------------------------------------
    # Locate the root page (IT Trouble Shooting Hub) either via main_id or by title.
    # ----------------------------------------------------------------------------------
    root_page_id = None
    if main_id:
        found_id, object_type = notion_utils.find_page_or_database_by_id(
            notion, main_id
        )
        if found_id and object_type == "page":
            root_page_id = found_id

    if not root_page_id:
        root_page_id = notion_utils.find_page(notion, "IT Trouble Shooting Hub")

    if not sc.check("IT Trouble Shooting Hub page found", bool(root_page_id),
                    "Could not locate the 'IT Trouble Shooting Hub' page"):
        sc.fail_remaining([
            "IT Requests database found", "Expected ticket found",
            "Priority is High", "Due date is correct",
            "Correct number of bullet items", "All bullet items valid"
        ], "root page not found")
        return sc.summary()

    # ----------------------------------------------------------------------------------
    # Find the IT Requests database under the root page.
    # ----------------------------------------------------------------------------------
    requests_db_id = notion_utils.find_database_in_block(
        notion, root_page_id, "IT Requests"
    )
    if not sc.check("IT Requests database found", bool(requests_db_id),
                    "'IT Requests' database not found in the workspace"):
        sc.fail_remaining([
            "Expected ticket found", "Priority is High", "Due date is correct",
            "Correct number of bullet items", "All bullet items valid"
        ], "IT Requests database not found")
        return sc.summary()

    # ----------------------------------------------------------------------------------
    # Search for the expected ticket inside the IT Requests database.
    # ----------------------------------------------------------------------------------
    expected_title = "Quarterly Security Audit - Expired Assets Review"
    results = notion_utils.query_database(notion, requests_db_id).get("results", [])

    target_page = None
    for page in results:
        title_text = _get_title_text(page.get("properties", {}))
        if title_text == expected_title:
            target_page = page
            break

    if not sc.check("Expected ticket found", bool(target_page),
                    f"Ticket with title '{expected_title}' was not found "
                    "in 'IT Requests' database"):
        sc.fail_remaining([
            "Priority is High", "Due date is correct",
            "Correct number of bullet items", "All bullet items valid"
        ], "expected ticket not found")
        return sc.summary()

    props = target_page.get("properties", {})

    # ----------------------------------------------------------------------------------
    # Validate Priority property.
    # ----------------------------------------------------------------------------------
    priority_value = props.get("Priority", {}).get("select", {}).get("name")
    sc.check("Priority is High", priority_value == "High",
             f"Expected Priority 'High', found '{priority_value}'")

    # ----------------------------------------------------------------------------------
    # Validate Due date property.
    # ----------------------------------------------------------------------------------
    due_date_start = props.get("Due", {}).get("date", {}).get("start")
    expected_due_iso = "2023-06-22"
    sc.check("Due date is correct",
             bool(due_date_start) and due_date_start.startswith(expected_due_iso),
             f"Expected Due date '{expected_due_iso}', found '{due_date_start}'")

    # ----------------------------------------------------------------------------------
    # Validate the bulleted list contains the correct expired items in required format.
    # ----------------------------------------------------------------------------------
    page_id = target_page["id"]
    blocks = notion.blocks.children.list(block_id=page_id).get("results", [])
    bullet_texts = [
        notion_utils.get_block_plain_text(b)
        for b in blocks
        if b.get("type") == "bulleted_list_item"
    ]

    expected_items = {
        "192371-8910/54": "Computer Accessory",
        "32x11PIP": "Computer Accessory",
        "76x87PCY": "Laptop",
        "36x10PIQ": "Computer Accessory",
        "65XYQ/GB": "License",
    }

    if not sc.check("Correct number of bullet items",
                    len(bullet_texts) == len(expected_items),
                    f"Expected {len(expected_items)} bullet items, found {len(bullet_texts)}"):
        sc.fail_remaining(["All bullet items valid"], "wrong number of bullet items")
        return sc.summary()

    bullet_pattern = re.compile(r"^\s*(.*?)\s+-\s+(.*?)\s+-\s+(.+?)\s*$")
    matched = set()
    bullets_valid = True
    bullets_err = ""
    for text in bullet_texts:
        m = bullet_pattern.match(text)
        if not m:
            bullets_valid = False
            bullets_err = (f"Bullet item '{text}' does not follow "
                           "'<Serial> - <Tag> - <Recommendation>' format")
            break
        serial, tag, advice = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if serial not in expected_items:
            bullets_valid = False
            bullets_err = f"Unexpected Serial '{serial}' found in bullet list"
            break
        if expected_items[serial] != tag:
            expected_tag = expected_items[serial]
            bullets_valid = False
            bullets_err = f"Serial '{serial}' expected tag '{expected_tag}', found '{tag}'"
            break
        if not advice:
            bullets_valid = False
            bullets_err = f"Bullet item for Serial '{serial}' is missing a recommendation/advice"
            break
        matched.add(serial)

    if bullets_valid and len(matched) != len(expected_items):
        missing = set(expected_items.keys()) - matched
        bullets_valid = False
        bullets_err = f"Missing bullet items for serials: {', '.join(missing)}"

    sc.check("All bullet items valid", bullets_valid, bullets_err)

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
