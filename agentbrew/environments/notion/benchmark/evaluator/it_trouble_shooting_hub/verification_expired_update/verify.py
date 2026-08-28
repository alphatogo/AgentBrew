"""Verification module for Verification Expired Update task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from typing import Optional
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector

CALL_OUT_TEXT = "VERIFICATION EXPIRED - This page needs review and re-verification"
CALL_OUT_ICON = "⚠️"
CALL_OUT_COLOR = "red_background"
IT_HOMEPAGE_DB_TITLE = "IT Homepage"
IT_REQUESTS_DB_TITLE = "IT Requests"
REQUEST_TITLE = "Batch Verification Update Required"
PRIORITY_HIGH = "High"
STATUS_IN_PROGRESS = "In progress"


def _get_main_page_id(notion: Client, main_id: Optional[str]) -> Optional[str]:
    """Resolve the main page id starting from CLI arg or by title search."""
    if main_id:
        found_id, obj_type = notion_utils.find_page_or_database_by_id(notion, main_id)
        if found_id and obj_type == "page":
            return found_id
    # Fallback to title search (case-insensitive)
    return notion_utils.find_page(notion, "It Trouble Shooting Hub")


def _fetch_database_id(
    notion: Client, parent_page_id: str, db_title: str
) -> Optional[str]:
    """Locate a child database by title inside a given page."""
    return notion_utils.find_database_in_block(notion, parent_page_id, db_title)


def _expired_pages(notion: Client, db_id: str) -> list[dict]:
    """Return list of page objects with Verification.state == 'expired'."""
    # Query all pages (API max 100 per call). If many pages expected, iterate.
    results = notion_utils.query_database(notion, db_id).get("results", [])
    expired = []
    for page in results:
        verification_prop = page.get("properties", {}).get("Verification", {})
        state = verification_prop.get("verification", {}).get("state")
        # Skip the IT Inventory database entry
        title_prop = page.get("properties", {}).get("Page", {}).get("title", [])
        title_text = title_prop[0].get("plain_text") if title_prop else ""
        if title_text.strip().lower() == "it inventory":
            continue

        if state and "expired" in state.lower():
            expired.append(page)
    return expired


def _check_callout_present(notion: Client, page_id: str) -> bool:
    """Verify the specified callout is the first child block of the page."""
    children = notion.blocks.children.list(block_id=page_id, page_size=1).get(
        "results", []
    )
    if not children:
        return False
    first_block = children[0]
    if first_block.get("type") != "callout":
        return False
    data = first_block.get("callout", {})
    # Check color
    if data.get("color") != CALL_OUT_COLOR:
        return False

    # Check icon
    icon = data.get("icon", {})
    if icon.get("type") != "emoji" or icon.get("emoji") != CALL_OUT_ICON:
        return False

    # Check text content (callout rich text plain text)
    plain_text = notion_utils.get_block_plain_text(first_block)
    return CALL_OUT_TEXT in plain_text


def _find_request_page(notion: Client, db_id: str) -> Optional[dict]:
    """Find the IT Request page with the expected title."""
    # Use a simple search inside database
    res = notion_utils.query_database(
        notion,
        db_id,
        filter={"property": "Task name", "title": {"equals": REQUEST_TITLE}},
    ).get("results", [])
    return res[0] if res else None


def _check_request_properties(page: dict) -> bool:
    props = page.get("properties", {})
    priority = props.get("Priority", {}).get("select", {}).get("name")
    status = (
        props.get("Status", {}).get("status", {}).get("name")
        if props.get("Status", {}).get("status")
        else props.get("Status", {}).get("select", {}).get("name")
    )
    return priority == PRIORITY_HIGH and status == STATUS_IN_PROGRESS


def _request_page_contains_mentions(
    notion: Client, request_page_id: str, expected_page_ids: list[str]
) -> bool:
    children = notion.blocks.children.list(block_id=request_page_id, page_size=100).get(
        "results", []
    )
    bullet_blocks = [b for b in children if b.get("type") == "bulleted_list_item"]
    mentioned_ids: set[str] = set()
    for block in bullet_blocks:
        rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
        for rt in rich_text:
            if rt.get("type") == "mention":
                mention = rt.get("mention", {})
                if mention.get("type") == "page":
                    mentioned_ids.add(mention.get("page", {}).get("id"))
    if len(mentioned_ids) < len(expected_page_ids):
        return False
    return all(pid in mentioned_ids for pid in expected_page_ids)


def verify(notion: Client, main_id: Optional[str] = None) -> tuple[bool, str]:
    """Verify that verification expired callouts and tickets are correctly updated."""
    sc = SubCriteriaCollector("Verification Expired Update")

    main_page_id = _get_main_page_id(notion, main_id)
    if not sc.check("Main page found", bool(main_page_id),
                    "Could not locate the main page 'It Trouble Shooting Hub'"):
        sc.fail_remaining([
            "Required databases found", "Expired pages found",
            "All expired pages have correct callout",
            "IT Request ticket found", "Request has correct priority/status",
            "Request body mentions all affected pages"
        ], "main page not found")
        return sc.summary()

    # Locate required databases
    it_home_db_id = _fetch_database_id(notion, main_page_id, IT_HOMEPAGE_DB_TITLE)
    it_req_db_id = _fetch_database_id(notion, main_page_id, IT_REQUESTS_DB_TITLE)

    if not sc.check("Required databases found", bool(it_home_db_id and it_req_db_id),
                    "Required databases not found under the main page"):
        sc.fail_remaining([
            "Expired pages found", "All expired pages have correct callout",
            "IT Request ticket found", "Request has correct priority/status",
            "Request body mentions all affected pages"
        ], "required databases not found")
        return sc.summary()

    # Identify expired pages
    expired_pages = _expired_pages(notion, it_home_db_id)
    if not sc.check("Expired pages found", bool(expired_pages),
                    "No expired pages found; expected at least one for this task"):
        sc.fail_remaining([
            "All expired pages have correct callout",
            "Request body mentions all affected pages"
        ], "no expired pages found")
        # Still check request ticket (independent)
    else:
        # Verify callout on each expired page
        callout_ok = True
        callout_err = ""
        for pg in expired_pages:
            pid = pg["id"]
            if not _check_callout_present(notion, pid):
                print(f"Failure: Callout missing or incorrect on page {pid}.", file=sys.stderr)
                callout_ok = False
                callout_err = f"Callout missing or incorrect on page {pid}"
        sc.check("All expired pages have correct callout", callout_ok, callout_err)

    # Verify IT Request entry (independent of callout check)
    request_page = _find_request_page(notion, it_req_db_id)
    if not sc.check("IT Request ticket found", bool(request_page),
                    "IT Request 'Batch Verification Update Required' not found"):
        sc.fail_remaining([
            "Request has correct priority/status",
            "Request body mentions all affected pages"
        ], "request ticket not found")
        return sc.summary()

    sc.check("Request has correct priority/status",
             _check_request_properties(request_page),
             "Priority or Status incorrect on IT Request")

    # Verify bullet list in IT Request body
    expected_page_ids = [p["id"] for p in expired_pages]
    sc.check("Request body mentions all affected pages",
             _request_page_contains_mentions(notion, request_page["id"], expected_page_ids),
             "IT Request body does not contain mentions for all affected pages")

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
