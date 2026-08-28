"""Verification module for Projects Section Update task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """
    Verifies that the projects section has been reorganized correctly with cross-section references.
    """
    sc = SubCriteriaCollector("Projects Section Update")

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
            "Projects database found", "Skills database found",
            "Highest skill found", "Knitties eComm Website deleted",
            "Zapier Dashboard Redesign exists", "Projects DB block found",
            "Divider after Projects DB", "Current Focus heading present",
            "Current Focus paragraph correct"
        ], "main page not found")
        return sc.summary()

    # Find the Projects database
    projects_db_id = notion_utils.find_database_in_block(notion, page_id, "Projects")
    if not sc.check("Projects database found", bool(projects_db_id),
                    "Database 'Projects' not found"):
        sc.fail_remaining([
            "Knitties eComm Website deleted", "Zapier Dashboard Redesign exists",
            "Projects DB block found", "Divider after Projects DB",
            "Current Focus heading present", "Current Focus paragraph correct"
        ], "Projects database not found")

    # Find the Skills database to get the highest skill level
    skills_db_id = notion_utils.find_database_in_block(notion, page_id, "Skills")
    if not sc.check("Skills database found", bool(skills_db_id),
                    "Database 'Skills' not found"):
        sc.fail_remaining([
            "Highest skill found",
            "Current Focus paragraph correct"
        ], "Skills database not found")

    # Query Skills database to find the highest skill level
    highest_skill_name = ""
    highest_skill_level = 0
    if skills_db_id:
        skills_results = notion_utils.query_database(notion, skills_db_id).get("results", [])

        for skill_page in skills_results:
            properties = skill_page.get("properties", {})
            skill_name_prop = properties.get("Skill", {}).get("title", [])
            skill_level_prop = properties.get("Skill Level", {}).get("number")

            if skill_name_prop and skill_level_prop is not None:
                skill_name = skill_name_prop[0].get("text", {}).get("content", "")
                if skill_level_prop > highest_skill_level:
                    highest_skill_level = skill_level_prop
                    highest_skill_name = skill_name

        sc.check("Highest skill found", bool(highest_skill_name),
                 "Could not find any skills with skill levels")

    if not projects_db_id:
        return sc.summary()

    # Query Projects database
    projects_results = notion_utils.query_database(notion, projects_db_id).get(
        "results", []
    )

    # Check that "Knitties eComm Website" is deleted
    knitties_deleted = True
    for page in projects_results:
        properties = page.get("properties", {})
        name_prop = properties.get("Name", {}).get("title", [])
        if (
            name_prop
            and name_prop[0].get("text", {}).get("content") == "Knitties eComm Website"
        ):
            print("Failure: 'Knitties eComm Website' project was not deleted.",
                  file=sys.stderr)
            knitties_deleted = False

    sc.check("Knitties eComm Website deleted", knitties_deleted,
             "'Knitties eComm Website' project was not deleted")

    # Check that "Zapier Dashboard Redesign" exists with correct properties
    zapier_project_found = False
    zapier_ok = True
    zapier_err = ""
    for page in projects_results:
        properties = page.get("properties", {})
        name_prop = properties.get("Name", {}).get("title", [])
        if (
            name_prop
            and name_prop[0].get("text", {}).get("content")
            == "Zapier Dashboard Redesign"
        ):
            zapier_project_found = True

            # Check description contains reference to UI Design Internship
            desc_prop = properties.get("Description", {}).get("rich_text", [])
            if not desc_prop:
                print("Failure: Zapier project has no description.", file=sys.stderr)
                zapier_ok = False
                zapier_err = "Zapier project has no description"
                break

            description_text = desc_prop[0].get("text", {}).get("content", "")
            base_desc = (
                "Led the complete redesign of Zapier's main dashboard, "
                "focusing on improved usability and modern design patterns. "
                "Implemented new navigation system and responsive layouts."
            )
            if base_desc not in description_text:
                print("Failure: Zapier project description is missing base content.",
                      file=sys.stderr)
                zapier_ok = False
                zapier_err = "Zapier project description is missing base content"
                break

            # Check date
            date_prop = properties.get("Date", {}).get("date", {})
            if (
                not date_prop
                or date_prop.get("start") != "2024-01-01"
                or date_prop.get("end") != "2024-06-30"
            ):
                print("Failure: Zapier project date range is incorrect.", file=sys.stderr)
                zapier_ok = False
                zapier_err = "Zapier project date range is incorrect"
                break

            # Check tags
            tags_prop = properties.get("Tags", {}).get("multi_select", [])
            tag_names = {tag.get("name") for tag in tags_prop}
            if "UI Design" not in tag_names or "Enterprise" not in tag_names:
                print("Failure: Zapier project is missing required tags.", file=sys.stderr)
                zapier_ok = False
                zapier_err = "Zapier project is missing required tags"
                break

            # Check phone
            phone_prop = properties.get("Phone", {}).get("phone_number", [])
            if not phone_prop or phone_prop != "+44 7871263013":
                print("Failure: Zapier project phone number is incorrect.", file=sys.stderr)
                zapier_ok = False
                zapier_err = "Zapier project phone number is incorrect"
                break

            # Check url
            url_prop = properties.get("Url", {}).get("url", [])
            if not url_prop or url_prop != "www.zinenwine.com":
                print("Failure: Zapier project url is incorrect.", file=sys.stderr)
                zapier_ok = False
                zapier_err = "Zapier project url is incorrect"
                break

            # Check Enterprise tag color
            enterprise_tag_purple = False
            for tag in tags_prop:
                if tag.get("name") == "Enterprise" and tag.get("color") == "purple":
                    enterprise_tag_purple = True
                    break
            if not enterprise_tag_purple:
                print("Failure: Enterprise tag does not have purple color.", file=sys.stderr)
                zapier_ok = False
                zapier_err = "Enterprise tag does not have purple color"

            break

    if not zapier_project_found:
        sc.check("Zapier Dashboard Redesign exists", False,
                 "'Zapier Dashboard Redesign' project not found")
    else:
        sc.check("Zapier Dashboard Redesign exists", zapier_ok, zapier_err)

    # Find the Projects database block and verify blocks after it
    all_blocks = notion_utils.get_all_blocks_recursively(notion, page_id)

    # Find the Projects database block
    projects_db_index = -1
    for i, block in enumerate(all_blocks):
        if (
            block.get("type") == "child_database"
            and block.get("child_database", {}).get("title") == "Projects"
        ):
            projects_db_index = i
            break

    if not sc.check("Projects DB block found", projects_db_index != -1,
                    "Could not find Projects database block"):
        sc.fail_remaining([
            "Divider after Projects DB", "Current Focus heading present",
            "Current Focus paragraph correct"
        ], "Projects DB block not found")
        return sc.summary()

    # Check blocks after Projects database
    if not sc.check("Enough blocks after Projects DB",
                    projects_db_index + 3 <= len(all_blocks),
                    "Not enough blocks after Projects database"):
        sc.fail_remaining([
            "Divider after Projects DB", "Current Focus heading present",
            "Current Focus paragraph correct"
        ], "not enough blocks")
        return sc.summary()

    # Check divider block
    divider_block = all_blocks[projects_db_index + 1]
    sc.check("Divider after Projects DB", divider_block.get("type") == "divider",
             "Expected divider block after Projects database")

    # Check heading block
    heading_block = all_blocks[projects_db_index + 2]
    heading_ok = heading_block.get("type") == "heading_2"
    if heading_ok:
        heading_text = heading_block.get("heading_2", {}).get("rich_text", [])
        heading_ok = (
            bool(heading_text)
            and heading_text[0].get("text", {}).get("content") == "Current Focus"
        )
    sc.check("Current Focus heading present", heading_ok,
             "Expected heading_2 block with text 'Current Focus' after divider")

    # Check paragraph block with dynamic skill reference
    paragraph_block = all_blocks[projects_db_index + 3]
    para_ok = paragraph_block.get("type") == "paragraph"
    para_err = ""
    if para_ok:
        paragraph_text = paragraph_block.get("paragraph", {}).get("rich_text", [])
        if not paragraph_text:
            para_ok = False
            para_err = "Paragraph block is empty"
        else:
            paragraph_content = paragraph_text[0].get("text", {}).get("content", "")

            # Check that paragraph contains the base text
            base_text = (
                "The Zapier Dashboard Redesign represents my most impactful recent "
                "work, leveraging my expertise in"
            )
            if base_text not in paragraph_content:
                para_ok = False
                para_err = "Paragraph does not contain base text"
            elif highest_skill_name:
                # Check that paragraph references the highest skill
                skill_level_percent = int(highest_skill_level * 100)
                expected_skill_ref = f"{highest_skill_name} ({skill_level_percent}%)"
                if expected_skill_ref not in paragraph_content:
                    para_ok = False
                    para_err = (f"Paragraph does not reference highest skill "
                                f"'{expected_skill_ref}'")
            else:
                # Check that paragraph contains the ending text
                ending_text = (
                    "enterprise-grade solutions that prioritize both aesthetics and functionality"
                )
                if ending_text not in paragraph_content:
                    para_ok = False
                    para_err = "Paragraph does not contain proper ending text"
    else:
        para_err = "Expected paragraph block after heading"

    sc.check("Current Focus paragraph correct", para_ok, para_err)

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
