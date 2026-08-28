"""Verification module for Goals Restructure task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from typing import List
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector

# Expected new goal heading text (including emoji)
NEW_GOAL_HEADING = "🔄 Digital Transformation Initiative"

# Section title to look for
GOALS_SECTION_TITLE = "Current Goals"


def _plain(block) -> str:
    """Return concatenated plain text of a block."""
    return notion_utils.get_block_plain_text(block)


# Some Notion rich-text strings may include non-breaking spaces (\xa0) after emoji.
# Normalize them to plain spaces so text matching is robust.
def _normalize_string(s: str) -> str:
    return s.replace("\xa0", " ")


def _is_heading(block) -> bool:
    return block.get("type") in ["heading_1", "heading_2", "heading_3"]


def _is_toggle(block) -> bool:
    """Determine whether a block is a toggle (standard toggle block or toggle-able heading)."""
    btype = block.get("type")
    # In our scenario, goal blocks are headings (usually heading_3) marked as toggleable.
    if btype in ["heading_1", "heading_2", "heading_3"]:
        heading_data = block.get(btype, {})
        return heading_data.get("is_toggleable", False)
    # Some Notion pages may contain classic toggle blocks (type == "toggle"). They are
    # not expected in this task, but keeping this check allows broader compatibility.
    return btype == "toggle"


def _get_children(notion: Client, block_id: str) -> List[dict]:
    """Retrieve **direct** children of a block.

    No pagination handling needed for small test pages.
    """
    try:
        return notion.blocks.children.list(block_id=block_id).get("results", [])
    except (ValueError, KeyError, TypeError, AttributeError):
        return []


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """Verifies that the Company in a Box page has been updated per the task requirements."""
    sc = SubCriteriaCollector("Goals Restructure")

    # 1. Locate the main page
    page_id = None
    if main_id:
        found_id, obj_type = notion_utils.find_page_or_database_by_id(notion, main_id)
        if found_id and obj_type == "page":
            page_id = found_id

    if not page_id:
        # Try a few case variations just in case
        for title in [
            "Company In A Box",
        ]:
            page_id = notion_utils.find_page(notion, title)
            if page_id:
                break

    if not sc.check("Company In A Box page found", bool(page_id),
                    "Could not find the 'Company in a Box' page"):
        sc.fail_remaining([
            "Current Goals heading found", "Goals section non-empty",
            "Exactly 4 toggle blocks", "New goal heading present",
            "Each toggle has content", "No non-toggle heading_3 blocks"
        ], "main page not found")
        return sc.summary()

    # 2. Recursively locate the "Current Goals" heading and collect its sibling blocks

    def _fetch_children(bid: str) -> List[dict]:
        try:
            return notion.blocks.children.list(block_id=bid).get("results", [])
        except (ValueError, KeyError, TypeError, AttributeError):
            return []

    goals_section_blocks: List[dict] = []

    # Breadth-first traversal to find the heading
    queue = [page_id]
    found_parent = None
    found_index = None

    while queue and found_parent is None:
        parent_id = queue.pop(0)
        children = _fetch_children(parent_id)
        for idx, child in enumerate(children):
            if (
                _is_heading(child)
                and GOALS_SECTION_TITLE.lower()
                in _normalize_string(_plain(child)).lower()
            ):
                found_parent = parent_id
                found_index = idx
                break
        # enqueue grandchildren for further search
        for ch in children:
            if ch.get("has_children"):
                queue.append(ch["id"])

    if not sc.check("Current Goals heading found", found_parent is not None,
                    "Could not find the 'Current Goals' heading anywhere in the page"):
        sc.fail_remaining([
            "Goals section non-empty", "Exactly 4 toggle blocks",
            "New goal heading present", "Each toggle has content",
            "No non-toggle heading_3 blocks"
        ], "Current Goals heading not found")
        return sc.summary()

    # Retrieve siblings once more to get the final list and slice after heading.
    siblings = _fetch_children(found_parent)
    if not sc.check("Goals section index valid",
                    found_index is not None and found_index < len(siblings),
                    "Internal logic issue when locating Current Goals section"):
        sc.fail_remaining([
            "Goals section non-empty", "Exactly 4 toggle blocks",
            "New goal heading present", "Each toggle has content",
            "No non-toggle heading_3 blocks"
        ], "invalid section index")
        return sc.summary()

    goals_section_blocks = siblings[found_index + 1:]

    if not sc.check("Goals section non-empty", bool(goals_section_blocks),
                    "'Current Goals' section appears to be empty"):
        sc.fail_remaining([
            "Exactly 4 toggle blocks", "New goal heading present",
            "Each toggle has content", "No non-toggle heading_3 blocks"
        ], "goals section is empty")
        return sc.summary()

    # 3. Identify toggle blocks that represent goals
    toggle_blocks = [b for b in goals_section_blocks if _is_toggle(b)]

    if not sc.check("Exactly 4 toggle blocks", len(toggle_blocks) == 4,
                    f"Expected 4 toggle blocks for goals, found {len(toggle_blocks)}"):
        sc.fail_remaining([
            "New goal heading present", "Each toggle has content",
            "No non-toggle heading_3 blocks"
        ], "wrong number of toggle blocks")
        return sc.summary()

    # 4. Ensure the new goal heading exists among the toggles
    found_new_goal = any(
        _normalize_string(NEW_GOAL_HEADING).lower()
        in _normalize_string(_plain(tb)).lower()
        for tb in toggle_blocks
    )
    sc.check("New goal heading present", found_new_goal,
             f"Did not find a toggle block with heading '{NEW_GOAL_HEADING}'")

    # 5. Validate that each toggle has at least one child paragraph/description
    content_types = {
        "paragraph",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
        "callout",
        "quote",
    }
    toggles_with_content_ok = True
    for tb in toggle_blocks:
        if (
            _normalize_string(NEW_GOAL_HEADING).lower()
            in _normalize_string(_plain(tb)).lower()
        ):
            # Skip checking the new goal itself, as it does not have a description yet.
            continue
        if not tb.get("has_children", False):
            toggle_name = _normalize_string(_plain(tb))
            print(f"Error: Toggle '{toggle_name}' has no child blocks "
                  "(description not moved).", file=sys.stderr)
            toggles_with_content_ok = False
            continue
        children = _get_children(notion, tb["id"])
        if not any(c.get("type") in content_types for c in children):
            toggle_name = _normalize_string(_plain(tb))
            print(f"Error: Toggle '{toggle_name}' seems to lack any "
                  "description/content inside it.", file=sys.stderr)
            toggles_with_content_ok = False

    sc.check("Each toggle has content", toggles_with_content_ok,
             "One or more toggle blocks are missing description content")

    # 6. Confirm that there are no residual heading_3 blocks (non-toggle) for the goals
    non_toggle_headings = [
        b
        for b in goals_section_blocks
        if b.get("type") == "heading_3" and not _is_toggle(b)
    ]
    if non_toggle_headings:
        titles = [_normalize_string(_plain(b)) for b in non_toggle_headings]
        sc.check("No non-toggle heading_3 blocks", False,
                 f"Found heading_3 blocks not converted to toggles: {titles}")
    else:
        sc.check("No non-toggle heading_3 blocks", True)

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
