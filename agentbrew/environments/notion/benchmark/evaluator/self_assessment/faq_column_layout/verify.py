"""Verification module for Faq Column Layout task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-return-statements,too-many-statements
    """
    Verifies that the FAQ toggle has been properly reorganized with a column list.
    """
    sc = SubCriteriaCollector("Faq Column Layout")

    # Start from main_id if provided
    page_id = None
    if main_id:
        found_id, object_type = notion_utils.find_page_or_database_by_id(
            notion, main_id
        )
        if found_id and object_type == "page":
            page_id = found_id

    if not page_id:
        # Try to find the Self Assessment page
        page_id = notion_utils.find_page(notion, "Self Assessment")

    if not sc.check("Self Assessment page exists", bool(page_id), "Self Assessment page not found."):
        sc.fail_remaining(
            [
                "FAQ toggle exists",
                "FAQ contains a column list",
                "No Q&A content outside column list",
                "Exactly two columns exist",
                "Column 1 has 2 Q&A pairs",
                "Column 2 has 2 Q&A pairs",
            ],
            "page not found",
        )
        return sc.summary()

    # Get all blocks recursively from the page
    all_blocks = notion_utils.get_all_blocks_recursively(notion, page_id)

    # Find the FAQ toggle block
    faq_toggle_block = None
    faq_toggle_id = None
    for block in all_blocks:
        if block.get("type") == "toggle":
            block_text = notion_utils.get_block_plain_text(block)
            if "FAQ" in block_text:
                faq_toggle_block = block
                faq_toggle_id = block.get("id")
                print(f"Found FAQ toggle block: {block_text}")
                break

    if not sc.check("FAQ toggle exists", bool(faq_toggle_block), "FAQ toggle block not found."):
        sc.fail_remaining(
            [
                "FAQ contains a column list",
                "No Q&A content outside column list",
                "Exactly two columns exist",
                "Column 1 has 2 Q&A pairs",
                "Column 2 has 2 Q&A pairs",
            ],
            "FAQ toggle not found",
        )
        return sc.summary()

    # Find column_list inside the FAQ toggle
    column_list_block = None
    for block in all_blocks:
        if (
            block.get("type") == "column_list"
            and block.get("parent", {}).get("block_id") == faq_toggle_id
        ):
            column_list_block = block
            break

    if not sc.check(
        "FAQ contains a column list",
        bool(column_list_block),
        "No column_list found inside FAQ toggle.",
    ):
        sc.fail_remaining(
            [
                "No Q&A content outside column list",
                "Exactly two columns exist",
                "Column 1 has 2 Q&A pairs",
                "Column 2 has 2 Q&A pairs",
            ],
            "column list not found",
        )
        return sc.summary()

    # Check that there are no Q&A pairs directly under FAQ toggle (outside column_list)
    direct_faq_children = []
    for block in all_blocks:
        if block.get("parent", {}).get("block_id") == faq_toggle_id and block.get(
            "id"
        ) != column_list_block.get("id"):
            direct_faq_children.append(block)

    # Check if any of these are heading_3 or paragraph blocks (Q&A content)
    outside_qa_blocks = []
    for block in direct_faq_children:
        if block.get("type") in ["heading_3", "paragraph"]:
            block_text = notion_utils.get_block_plain_text(block)[:50]
            outside_qa_blocks.append(f"{block.get('type')}: {block_text}...")

    sc.check(
        "No Q&A content outside column list",
        not outside_qa_blocks,
        f"Found Q&A content outside column_list: {outside_qa_blocks}",
    )

    # Find the two columns
    columns = []
    column_list_id = column_list_block.get("id")
    for block in all_blocks:
        if (
            block.get("type") == "column"
            and block.get("parent", {}).get("block_id") == column_list_id
        ):
            columns.append(block)

    if not sc.check(
        "Exactly two columns exist",
        len(columns) == 2,
        f"Expected 2 columns, found {len(columns)}.",
    ):
        sc.fail_remaining(
            ["Column 1 has 2 Q&A pairs", "Column 2 has 2 Q&A pairs"],
            "column count mismatch",
        )
        return sc.summary()

    # Check each column has exactly 2 Q&A pairs
    for i, column in enumerate(columns):
        column_id = column.get("id")

        # Find blocks inside this column
        column_blocks = []
        for block in all_blocks:
            if block.get("parent", {}).get("block_id") == column_id:
                column_blocks.append(block)

        # Count Q&A pairs (should be heading_3 followed by paragraph)
        qa_pairs = 0
        j = 0
        while j < len(column_blocks):
            if (
                column_blocks[j].get("type") == "heading_3"
                and j + 1 < len(column_blocks)
                and column_blocks[j + 1].get("type") == "paragraph"
            ):
                qa_pairs += 1
                j += 2  # Skip both question and answer
            else:
                j += 1

        sc.check(
            f"Column {i + 1} has 2 Q&A pairs",
            qa_pairs == 2,
            f"Column {i + 1} has {qa_pairs} Q&A pairs, expected 2.",
        )

    return sc.summary()


def main():
    """
    Executes the verification process and exits with a status code.
    """
    notion = notion_utils.get_notion_client()
    main_id = sys.argv[1] if len(sys.argv) > 1 else None
    success, _error_msg = verify(notion, main_id)
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
