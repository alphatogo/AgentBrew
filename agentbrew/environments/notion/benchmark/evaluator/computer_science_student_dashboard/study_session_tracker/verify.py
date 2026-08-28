"""Verification module for Study Session Tracker task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from typing import Dict, Optional

from notion_client import Client

from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def _normalize_string(s: str) -> str:
    """Replace non-breaking space with regular space for safe comparison."""
    return s.replace("\xa0", " ")


def verify(notion: Client, main_id: Optional[str] = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """Verify that the new study-session entry for 2025-01-29 was added correctly.

    The script checks that:
    1. A bold date-mention with start=2025-01-29 exists.
    2. The mention sits after the 2022-09-02 section but before the divider that originally
       followed that section.
    3. Exactly four specified to-do items follow the new date mention and they are all unchecked.
    """
    sc = SubCriteriaCollector("Study Session Tracker")

    # ---------------------------------------------------------------------
    # Locate the main page -------------------------------------------------
    # ---------------------------------------------------------------------
    page_id: Optional[str] = None

    if main_id:
        found_id, object_type = notion_utils.find_page_or_database_by_id(
            notion, main_id
        )
        if found_id and object_type == "page":
            page_id = found_id

    if not page_id:
        page_id = notion_utils.find_page(notion, "Computer Science Student Dashboard")

    if not sc.check("Dashboard page found", bool(page_id),
                    "Page 'Computer Science Student Dashboard' not found"):
        sc.fail_remaining([
            "2022-09-02 date section found", "Divider after 2022-09-02 found",
            "2025-01-29 date mention found", "2025-01-29 date mention is bold",
            "2025-01-29 section positioned correctly", "All 4 todo items present"
        ], "page not found")
        return sc.summary()

    # ---------------------------------------------------------------------
    # Fetch all blocks under the page (flattened order) --------------------
    # ---------------------------------------------------------------------
    all_blocks = notion_utils.get_all_blocks_recursively(notion, page_id)

    # ---------------------------------------------------------------------
    # Locate reference blocks ---------------------------------------------
    # ---------------------------------------------------------------------
    TARGET_DATE = "2025-01-29"  # pylint: disable=invalid-name
    PREVIOUS_DATE = "2022-09-02"  # pylint: disable=invalid-name

    index_previous_date: Optional[int] = None
    index_new_date: Optional[int] = None
    index_divider_after_previous: Optional[int] = None
    new_date_bold = True

    for idx, block in enumerate(all_blocks):
        # Divider detection (we care only about the first divider that appears after
        # the 2022-09-02 block)
        if block.get("type") == "divider":
            if index_previous_date is not None and index_divider_after_previous is None:
                index_divider_after_previous = idx

        # We only need to inspect paragraph blocks that contain a date mention
        if block.get("type") != "paragraph":
            continue

        rich_text_list = block["paragraph"].get("rich_text", [])
        for rt in rich_text_list:
            if (
                rt.get("type") != "mention"
                or rt.get("mention", {}).get("type") != "date"
            ):
                continue

            date_start = rt["mention"]["date"].get("start")

            if date_start == PREVIOUS_DATE and index_previous_date is None:
                index_previous_date = idx

            if date_start == TARGET_DATE and index_new_date is None:
                index_new_date = idx
                # (1) Verify bold annotation
                if not rt.get("annotations", {}).get("bold", False):
                    new_date_bold = False

    sc.check("2022-09-02 date section found", index_previous_date is not None,
             "Could not locate the 2022-09-02 date section")
    sc.check("Divider after 2022-09-02 found", index_divider_after_previous is not None,
             "Could not locate the divider that follows the 2022-09-02 section")
    sc.check("2025-01-29 date mention found", index_new_date is not None,
             "Could not locate the new 2025-01-29 date mention")

    if index_new_date is not None:
        sc.check("2025-01-29 date mention is bold", new_date_bold,
                 "The 2025-01-29 date mention is not bold")

    # (2) Verify ordering — only if all indices are found
    if index_previous_date is not None and index_new_date is not None and index_divider_after_previous is not None:
        sc.check("2025-01-29 section positioned correctly",
                 index_previous_date < index_new_date < index_divider_after_previous,
                 "The 2025-01-29 section is positioned incorrectly")
    else:
        if index_new_date is not None:
            sc.check("2025-01-29 section positioned correctly", False,
                     "Could not verify positioning due to missing reference blocks")

    # ---------------------------------------------------------------------
    # Verify to-do items under the new date section ------------------------
    # ---------------------------------------------------------------------
    if index_new_date is not None and index_divider_after_previous is not None:
        expected_texts = [
            "🧠 Review algorithms for technical interview",
            "📚 Study database systems chapter 7",
            "⚡ Practice system design problems",
            "🎯 Complete data structures assignment",
        ]
        expected_todos: Dict[str, bool] = {
            _normalize_string(t): False for t in expected_texts
        }

        todos_checked_err = []
        # Look through the blocks that lie between the new date mention and the divider
        for block in all_blocks[index_new_date + 1: index_divider_after_previous]:
            if block.get("type") != "to_do":
                continue

            plain_text = notion_utils.get_block_plain_text(block).strip()
            plain_text_norm = _normalize_string(plain_text)
            if plain_text_norm in expected_todos:
                # (3a) Verify the to-do is unchecked
                if block["to_do"].get("checked", False):
                    todos_checked_err.append(plain_text)
                expected_todos[plain_text_norm] = True

        missing_items = [text for text, found in expected_todos.items() if not found]
        if todos_checked_err:
            sc.check("All 4 todo items present", False,
                     f"To-do items are checked (should be unchecked): {todos_checked_err}")
        elif missing_items:
            sc.check("All 4 todo items present", False,
                     f"Missing to-do items: {missing_items}")
        else:
            sc.check("All 4 todo items present", True)
    else:
        sc.check("All 4 todo items present", False,
                 "Could not verify todos due to missing reference blocks")

    return sc.summary()


# -------------------------------------------------------------------------
# Command-line entry-point -------------------------------------------------
# -------------------------------------------------------------------------


def main() -> None:
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
