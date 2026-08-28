"""Verification module for Section Organization task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector

def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-return-statements,too-many-statements
    """
    Verifies that the Standard Operating Procedure page has been reorganized correctly.
    """
    sc = SubCriteriaCollector("Section Organization")

    # Step 1: Find the Standard Operating Procedure page
    if main_id:
        found_id, object_type = notion_utils.find_page_or_database_by_id(notion, main_id)
        if not found_id or object_type != 'page':
            found_id = None
    else:
        # Try to find the page by searching
        found_id = notion_utils.find_page(notion, "Standard Operating Procedure")
    if not sc.check(
        "Standard Operating Procedure page exists",
        bool(found_id),
        "Standard Operating Procedure page not found.",
    ):
        sc.fail_remaining(
            [
                "Roles section exists",
                "Column layout exists",
                "Original pages toggle exists",
                "Procedure section exists",
                "Section order is correct",
                "Column list children can be retrieved",
                "Column layout has at least 2 columns",
                "Left column is a column block",
                "Tools heading is in left column",
                "Tools column has at least 2 link_to_page blocks",
                "Right column is a column block",
                "Terminologies heading is in right column",
                "Toggle children can be retrieved",
                "Notion child page exists in toggle",
                "Figma child page exists in toggle",
                "No top-level Terminologies before Roles",
                "No standalone Tools heading outside column layout",
            ],
            "page not found",
        )
        return sc.summary()

    print(f"Found Standard Operating Procedure page: {found_id}")

    # Get all blocks from the page
    all_blocks = notion_utils.get_all_blocks_recursively(notion, found_id)
    print(f"Found {len(all_blocks)} blocks")

    print("Starting verification...")

    # Step 2: Verify the structure and section order
    print("2. Checking page structure and section order...")

    # Expected structure after the initial content and dividers
    # We'll look for main sections by their headings
    roles_index = None
    tools_column_index = None
    toggle_index = None
    procedure_index = None

    for i, block in enumerate(all_blocks):
        if block.get("type") == "heading_2":
            heading_text = ""
            rich_text = block.get("heading_2", {}).get("rich_text", [])
            if rich_text:
                heading_text = rich_text[0].get("text", {}).get("content", "")

            if heading_text == "Roles & responsibilities":
                roles_index = i
                print(f"✓ Found 'Roles & responsibilities' section at index {i}")
            elif heading_text == "Procedure":
                procedure_index = i
                print(f"✓ Found 'Procedure' section at index {i}")

    # Check for column_list (containing Tools and Terminologies)
    for i, block in enumerate(all_blocks):
        if block.get("type") == "column_list":
            # Check if this is the right column_list (should be after Roles & responsibilities)
            if roles_index and i > roles_index:
                tools_column_index = i
                print(f"✓ Found column_list at index {i}")
                break

    # Check for toggle block with "original pages"
    for i, block in enumerate(all_blocks):
        if block.get("type") == "toggle":
            toggle_text = ""
            rich_text = block.get("toggle", {}).get("rich_text", [])
            if rich_text:
                toggle_text = rich_text[0].get("text", {}).get("content", "")

            if toggle_text.lower() == "original pages":
                toggle_index = i
                print(f"✓ Found 'original pages' toggle at index {i}")
                break

    # Step 3: Verify section order
    print("3. Verifying section order...")

    roles_found = sc.check("Roles section exists", roles_index is not None, "'Roles & responsibilities' section not found.")
    column_layout_found = sc.check("Column layout exists", tools_column_index is not None, "Column layout not found.")
    toggle_found = sc.check("Original pages toggle exists", toggle_index is not None, "'original pages' toggle not found.")
    procedure_found = sc.check("Procedure section exists", procedure_index is not None, "'Procedure' section not found.")
    if not all([roles_found, column_layout_found, toggle_found, procedure_found]):
        sc.fail_remaining(
            [
                "Section order is correct",
                "Column list children can be retrieved",
                "Column layout has at least 2 columns",
                "Left column is a column block",
                "Tools heading is in left column",
                "Tools column has at least 2 link_to_page blocks",
                "Right column is a column block",
                "Terminologies heading is in right column",
                "Toggle children can be retrieved",
                "Notion child page exists in toggle",
                "Figma child page exists in toggle",
                "No top-level Terminologies before Roles",
                "No standalone Tools heading outside column layout",
            ],
            "required section missing",
        )
        return sc.summary()

    # Verify order: Roles & responsibilities < column_list < toggle < Procedure
    sc.check(
        "Section order is correct",
        roles_index < tools_column_index < toggle_index < procedure_index,
        (
            "Sections are not in the correct order. "
            f"Expected Roles ({roles_index}) < column_list ({tools_column_index}) "
            f"< toggle ({toggle_index}) < Procedure ({procedure_index})"
        ),
    )

    # Step 4: Verify column_list structure
    print("4. Verifying column layout structure...")

    column_list_block = all_blocks[tools_column_index]
    column_list_id = column_list_block.get("id")

    # Get direct children of column_list (should be columns only)
    column_children = []
    column_error = ""
    try:
        column_response = notion.blocks.children.list(block_id=column_list_id)
        column_children = column_response.get("results", [])
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        column_error = str(e)

    if not sc.check("Column list children can be retrieved", not column_error, f"Error getting column children: {column_error}"):
        sc.fail_remaining(
            [
                "Column layout has at least 2 columns",
                "Left column is a column block",
                "Tools heading is in left column",
                "Tools column has at least 2 link_to_page blocks",
                "Right column is a column block",
                "Terminologies heading is in right column",
            ],
            "column children unavailable",
        )
        return sc.summary()

    if not sc.check(
        "Column layout has at least 2 columns",
        len(column_children) >= 2,
        f"Column list should have at least 2 columns, found {len(column_children)}.",
    ):
        sc.fail_remaining(
            [
                "Left column is a column block",
                "Tools heading is in left column",
                "Tools column has at least 2 link_to_page blocks",
                "Right column is a column block",
                "Terminologies heading is in right column",
            ],
            "not enough columns",
        )
        return sc.summary()

    # Verify left column (Tools)
    left_column = column_children[0]
    if not sc.check(
        "Left column is a column block",
        left_column.get("type") == "column",
        "First child of column_list should be a column.",
    ):
        sc.fail_remaining(
            ["Tools heading is in left column", "Tools column has at least 2 link_to_page blocks"],
            "left column invalid",
        )
        return sc.summary()

    left_column_id = left_column.get("id")
    left_column_blocks = notion_utils.get_all_blocks_recursively(notion, left_column_id)

    # Check for Tools heading and link_to_page blocks in left column
    tools_heading_found = False
    link_to_page_count = 0
    for block in left_column_blocks:
        if block.get("type") == "heading_2":
            heading_data = block.get("heading_2", {}).get("rich_text", [{}])
            heading_text = heading_data[0].get("text", {}).get("content", "")
            if heading_text == "Tools":
                tools_heading_found = True
                print("✓ Found 'Tools' heading in left column")
        elif block.get("type") == "link_to_page":
            link_to_page_count += 1

    sc.check("Tools heading is in left column", tools_heading_found, "'Tools' heading not found in left column.")
    sc.check(
        "Tools column has at least 2 link_to_page blocks",
        link_to_page_count >= 2,
        f"Tools column should have at least 2 link_to_page blocks, found {link_to_page_count}.",
    )

    # Verify right column (Terminologies)
    right_column = column_children[1]
    if not sc.check(
        "Right column is a column block",
        right_column.get("type") == "column",
        "Second child of column_list should be a column.",
    ):
        sc.fail_remaining(["Terminologies heading is in right column"], "right column invalid")
        return sc.summary()

    right_column_id = right_column.get("id")
    right_column_blocks = notion_utils.get_all_blocks_recursively(notion, right_column_id)

    # Check for Terminologies heading in right column
    terminologies_heading_found = False
    for block in right_column_blocks:
        if block.get("type") == "heading_2":
            heading_data = block.get("heading_2", {}).get("rich_text", [{}])
            heading_text = heading_data[0].get("text", {}).get("content", "")
            if heading_text == "Terminologies":
                terminologies_heading_found = True
                print("✓ Found 'Terminologies' heading in right column")
                break

    sc.check(
        "Terminologies heading is in right column",
        terminologies_heading_found,
        "'Terminologies' heading not found in right column.",
    )

    # Step 5: Verify toggle block content
    print("5. Verifying toggle block content...")

    toggle_block = all_blocks[toggle_index]
    toggle_id = toggle_block.get("id")

    # Get direct children of toggle
    toggle_children = []
    toggle_error = ""
    try:
        toggle_response = notion.blocks.children.list(block_id=toggle_id)
        toggle_children = toggle_response.get("results", [])
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        toggle_error = str(e)

    if not sc.check("Toggle children can be retrieved", not toggle_error, f"Error getting toggle children: {toggle_error}"):
        sc.fail_remaining(
            ["Notion child page exists in toggle", "Figma child page exists in toggle"],
            "toggle children unavailable",
        )
        return sc.summary()

    # Check for child_page blocks (Notion and Figma)
    notion_page_found = False
    figma_page_found = False

    for block in toggle_children:
        if block.get("type") == "child_page":
            title = block.get("child_page", {}).get("title", "")
            if title == "Notion":
                notion_page_found = True
                print("✓ Found 'Notion' child page in toggle")
            elif title == "Figma":
                figma_page_found = True
                print("✓ Found 'Figma' child page in toggle")

    sc.check("Notion child page exists in toggle", notion_page_found, "'Notion' child page not found in toggle block.")
    sc.check("Figma child page exists in toggle", figma_page_found, "'Figma' child page not found in toggle block.")

    # Step 6: Verify that original sections no longer exist at top level
    print("6. Verifying original sections have been removed from top level...")

    # Check that there's no standalone "Terminologies" heading before
    # "Roles & responsibilities"
    terminology_before_roles = False
    for i in range(0, roles_index if roles_index else len(all_blocks)):
        block = all_blocks[i]
        if block.get("type") == "heading_2":
            heading_data = block.get("heading_2", {}).get("rich_text", [{}])
            heading_text = heading_data[0].get("text", {}).get("content", "")
            if heading_text == "Terminologies":
                terminology_before_roles = True
                break

    sc.check(
        "No top-level Terminologies before Roles",
        not terminology_before_roles,
        "'Terminologies' section found before 'Roles & responsibilities'.",
    )

    # Check that there's no standalone "Tools" heading outside the column
    tools_outside_column = False
    for i, block in enumerate(all_blocks):
        if i == tools_column_index:
            continue  # Skip the column_list itself
        if block.get("type") == "heading_2":
            heading_data = block.get("heading_2", {}).get("rich_text", [{}])
            heading_text = heading_data[0].get("text", {}).get("content", "")
            if heading_text == "Tools" and i != tools_column_index:
                # Check if this is NOT inside the column
                parent_id = block.get("parent", {}).get("block_id")
                if parent_id != left_column_id:
                    tools_outside_column = True
                    break

    sc.check(
        "No standalone Tools heading outside column layout",
        not tools_outside_column,
        "Standalone 'Tools' section found outside column layout.",
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
