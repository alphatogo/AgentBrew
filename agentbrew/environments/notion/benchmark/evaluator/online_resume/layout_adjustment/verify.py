"""Verification module for Layout Adjustment task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """
    Verifies that the Skills display has been adjusted correctly:
    1. Skills database on the right side should be deleted
    2. Skills section should be added on the left side under Languages
    3. Skills should be formatted with correct icons based on skill level
    4. Work History and Education sections should use black placeholder images
    """
    sc = SubCriteriaCollector("Layout Adjustment")

    page_id = None
    if main_id:
        found_id, object_type = notion_utils.find_page_or_database_by_id(
            notion, main_id
        )
        if found_id and object_type == "page":
            page_id = found_id

    if not page_id:
        page_id = notion_utils.find_page(notion, "Online Resume")

    if not sc.check("Online Resume page found", bool(page_id),
                    "Page 'Online Resume' not found"):
        sc.fail_remaining([
            "Skills database not in right column",
            "Main column list found", "Left column found", "Languages heading found",
            "Skills section found under Languages", "All expected skills present",
            "Work History images found", "Education images found"
        ], "main page not found")
        return sc.summary()

    all_blocks = notion_utils.get_all_blocks_recursively(notion, page_id)

    # Step 1: Verify Skills database is NOT in the right column anymore
    skills_still_in_right = False
    for block in all_blocks:  # pylint: disable=too-many-nested-blocks
        if block.get("type") == "column_list":
            column_list_id = block["id"]
            columns = notion_utils.get_all_blocks_recursively(notion, column_list_id)

            # Check if this is the main two-column layout
            if len(columns) == 2:
                # Find the right column (usually the one with larger width ratio)
                for column in columns:
                    if column.get("type") == "column":
                        width_ratio = column.get("column", {}).get("width_ratio", 0)
                        # Right column typically has width_ratio > 0.5
                        if width_ratio > 0.5:
                            right_column_id = column["id"]
                            right_column_blocks = notion_utils.get_all_blocks_recursively(
                                notion, right_column_id
                            )

                            # Check if Skills database exists in right column
                            for right_block in right_column_blocks:
                                if right_block.get("type") == "child_database":
                                    db_title = (right_block.get("child_database", {})
                                                .get("title"))
                                    if db_title == "Skills":
                                        skills_still_in_right = True

    sc.check("Skills database not in right column", not skills_still_in_right,
             "Skills database still exists in the right column")

    # Step 2: Find the left column and verify Skills section exists there
    skills_section_found = False
    skills_with_double_sparkles = []
    skills_with_single_sparkle = []
    skills_err = ""

    # First, find the main column_list (top-level)
    main_column_list_id = None
    for block in all_blocks:
        if block.get("type") == "column_list" and block.get("parent", {}).get("type") == "page_id":
            main_column_list_id = block["id"]
            break

    if not sc.check("Main column list found", bool(main_column_list_id),
                    "Main column list not found"):
        sc.fail_remaining([
            "Left column found", "Languages heading found",
            "Skills section found under Languages", "All expected skills present"
        ], "main column list not found")
    else:
        # Get the columns directly
        columns = notion_utils.get_all_blocks_recursively(notion, main_column_list_id)

        # Find the left column (the one with width_ratio around 0.25)
        left_column_id = None
        for column in columns:
            if column.get("type") == "column":
                width_ratio = column.get("column", {}).get("width_ratio", 0)
                # Left column has width_ratio around 0.25
                if 0.2 <= width_ratio <= 0.3:
                    left_column_id = column["id"]
                    break

        if not sc.check("Left column found", bool(left_column_id),
                        "Left column not found"):
            sc.fail_remaining([
                "Languages heading found",
                "Skills section found under Languages", "All expected skills present"
            ], "left column not found")
        else:
            # Get all blocks in the left column
            left_column_blocks = notion_utils.get_all_blocks_recursively(notion, left_column_id)

            # Find Languages heading
            languages_index = -1
            for i, left_block in enumerate(left_column_blocks):
                if (
                    left_block.get("type") == "heading_2"
                    and "Languages" in notion_utils.get_block_plain_text(left_block)
                ):
                    languages_index = i
                    break

            if not sc.check("Languages heading found", languages_index != -1,
                            "Languages heading not found in left column"):
                sc.fail_remaining([
                    "Skills section found under Languages", "All expected skills present"
                ], "Languages heading not found")
            else:
                # Look for Skills heading after Languages
                for i in range(languages_index + 1, len(left_column_blocks)):  # pylint: disable=too-many-nested-blocks
                    left_block = left_column_blocks[i]

                    if (
                        left_block.get("type") == "heading_2"
                        and "Skills" in notion_utils.get_block_plain_text(left_block)
                    ):
                        skills_section_found = True

                        # Check divider after Skills heading
                        if i + 1 < len(left_column_blocks):
                            next_block = left_column_blocks[i + 1]
                            if next_block.get("type") != "divider":
                                print("Error: Divider not found after Skills heading.",
                                      file=sys.stderr)
                                skills_err = "Divider not found after Skills heading"

                        # Collect skills after divider
                        for j in range(i + 2, len(left_column_blocks)):
                            skill_block = left_column_blocks[j]
                            if skill_block.get("type") == "paragraph":
                                skill_text = notion_utils.get_block_plain_text(skill_block)
                                if skill_text and skill_text.strip():  # Check for non-empty text
                                    # Check if text is bold
                                    rich_text = skill_block.get("paragraph", {}).get("rich_text", [])
                                    if rich_text and not rich_text[0].get("annotations", {}).get("bold"):
                                        print(f"Error: Skill '{skill_text}' is not bold.",
                                              file=sys.stderr)
                                        skills_err = f"Skill '{skill_text}' is not bold"

                                    # Check icon format
                                    if skill_text.startswith("✨✨"):
                                        skills_with_double_sparkles.append(skill_text)
                                    elif skill_text.startswith("✨"):
                                        skills_with_single_sparkle.append(skill_text)
                                    else:
                                        print(f"Error: Skill '{skill_text}' doesn't start with sparkle icon.",
                                              file=sys.stderr)
                                        skills_err = f"Skill '{skill_text}' doesn't start with sparkle icon"

                                    # Check format includes type in parentheses
                                    if "(" not in skill_text or ")" not in skill_text:
                                        print(f"Error: Skill '{skill_text}' doesn't include type in parentheses.",
                                              file=sys.stderr)
                                        skills_err = f"Skill '{skill_text}' doesn't include type in parentheses"
                            elif skill_block.get("type") in ["heading_1", "heading_2", "heading_3"]:
                                # Stop when we reach another section
                                break
                        break

                sc.check("Skills section found under Languages", skills_section_found,
                         skills_err if skills_err else "Skills section not found in the left column under Languages")

    # Step 3: Verify we have the expected skills
    expected_double_sparkle_skills = [
        "Photoshop",
        "Figma",
        "Notion",
        "Framer"
    ]

    expected_single_sparkle_skills = [
        "Webflow",
        "Rive",
        "CSS + Basic JS"
    ]

    skills_present_ok = True
    skills_present_err = ""
    # Check if all expected skills are present
    for skill_name in expected_double_sparkle_skills:
        found = any(skill_name in skill for skill in skills_with_double_sparkles)
        if not found:
            print(f"Error: Expected skill '{skill_name}' with ✨✨ not found.",
                  file=sys.stderr)
            skills_present_ok = False
            skills_present_err = f"Expected skill '{skill_name}' with ✨✨ not found"

    for skill_name in expected_single_sparkle_skills:
        found = any(skill_name in skill for skill in skills_with_single_sparkle)
        if not found:
            print(f"Error: Expected skill '{skill_name}' with ✨ not found.",
                  file=sys.stderr)
            skills_present_ok = False
            skills_present_err = f"Expected skill '{skill_name}' with ✨ not found"

    sc.check("All expected skills present", skills_present_ok, skills_present_err)

    # Step 4: Verify Work History and Education sections have black placeholder images
    work_history_images_found = 0
    education_images_found = 0
    black_placeholder_url = "https://singlecolorimage.com/get/000000/"

    # Find Work History and Education sections in the right column
    right_column_id = None
    main_column_list_id_2 = None
    for block in all_blocks:
        if block.get("type") == "column_list" and block.get("parent", {}).get("type") == "page_id":
            main_column_list_id_2 = block["id"]
            break

    if main_column_list_id_2:
        columns = notion_utils.get_all_blocks_recursively(notion, main_column_list_id_2)
        for column in columns:
            if column.get("type") == "column":
                width_ratio = column.get("column", {}).get("width_ratio", 0.5)
                # Right column has width_ratio around 0.75 or no width_ratio (which means equal split)
                if width_ratio > 0.6 or width_ratio == 0.5:
                    right_column_id = column["id"]
                    break

    if right_column_id:  # pylint: disable=too-many-nested-blocks
        right_column_blocks = notion_utils.get_all_blocks_recursively(notion, right_column_id)

        # Find Work History section
        work_history_index = -1
        education_index = -1

        for i, block in enumerate(right_column_blocks):
            if block.get("type") == "heading_1":
                heading_text = notion_utils.get_block_plain_text(block)
                if "Work History" in heading_text:
                    work_history_index = i
                elif "Education" in heading_text:
                    education_index = i

        # Check Work History column lists for images
        if work_history_index != -1:
            end_index = (education_index if education_index > work_history_index
                         else len(right_column_blocks))
            max_index = min(end_index, len(right_column_blocks))
            for i in range(work_history_index + 1, max_index):
                block = right_column_blocks[i]
                if block.get("type") == "column_list":
                    column_list_blocks = (
                        notion_utils.get_all_blocks_recursively(
                            notion, block["id"]
                        )
                    )
                    for column in column_list_blocks:
                        if column.get("type") == "column":
                            col_width = column.get("column", {}).get("width_ratio")
                            if col_width is None or col_width == 0.5:
                                column_contents = (
                                    notion_utils.get_all_blocks_recursively(
                                        notion, column["id"]
                                    )
                                )
                                for content_block in column_contents:
                                    if content_block.get("type") == "embed":
                                        embed_url = content_block.get("embed", {}).get("url", "")
                                        if black_placeholder_url in embed_url:
                                            work_history_images_found += 1
                                    elif content_block.get("type") == "image":
                                        image_data = content_block.get("image", {})
                                        image_url = image_data.get("external", {}).get("url", "")
                                        if black_placeholder_url in image_url:
                                            work_history_images_found += 1
                                break  # Only check first column

        # Check Education column list for images
        if education_index != -1:
            for i in range(education_index + 1, len(right_column_blocks)):
                block = right_column_blocks[i]
                if block.get("type") == "heading_1":
                    break  # Stop at next section
                if block.get("type") == "column_list":
                    column_list_blocks = (
                        notion_utils.get_all_blocks_recursively(
                            notion, block["id"]
                        )
                    )
                    for column in column_list_blocks:
                        if column.get("type") == "column":
                            col_width = column.get("column", {}).get("width_ratio")
                            if col_width is None or col_width == 0.5:
                                column_contents = (
                                    notion_utils.get_all_blocks_recursively(
                                        notion, column["id"]
                                    )
                                )
                                for content_block in column_contents:
                                    if content_block.get("type") == "embed":
                                        embed_url = content_block.get("embed", {}).get("url", "")
                                        if black_placeholder_url in embed_url:
                                            education_images_found += 1
                                    elif content_block.get("type") == "image":
                                        image_data = content_block.get("image", {})
                                        image_url = image_data.get("external", {}).get("url", "")
                                        if black_placeholder_url in image_url:
                                            education_images_found += 1
                                break  # Only check first column
                    break  # Only check first column_list in Education

    sc.check("Work History images found", work_history_images_found >= 2,
             f"Expected at least 2 Work History images with black placeholder, "
             f"found {work_history_images_found}")
    sc.check("Education images found", education_images_found >= 1,
             f"Expected at least 1 Education image with black placeholder, "
             f"found {education_images_found}")

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
