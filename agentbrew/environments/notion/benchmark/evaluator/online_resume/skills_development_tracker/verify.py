"""Verification module for Skills Development Tracker task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-return-statements,too-many-statements
    """
    Verifies that the Skills Development Tracker database and callout block were created correctly.
    """
    sc = SubCriteriaCollector("Skills Development Tracker")

    page_id = None
    if main_id:
        found_id, object_type = notion_utils.find_page_or_database_by_id(
            notion, main_id
        )
        if found_id and object_type == "page":
            page_id = found_id

    if not page_id:
        page_id = notion_utils.find_page(notion, "New Online Resume")

    if not sc.check("New Online Resume page found", bool(page_id),
                    "Page 'New Online Resume' not found"):
        sc.fail_remaining([
            "Skills Development Tracker database found",
            "Database schema valid",
            "Skills database found",
            "Tracker entries exist",
            "Entries have correct format",
            "Skills DB block found",
            "Callout found after Skills DB"
        ], "main page not found")
        return sc.summary()

    # Step 1: Verify Skills Development Tracker database exists
    tracker_db_id = notion_utils.find_database_in_block(
        notion, page_id, "Skills Development Tracker"
    )
    if not sc.check("Skills Development Tracker database found", bool(tracker_db_id),
                    "Database 'Skills Development Tracker' not found"):
        sc.fail_remaining([
            "Database schema valid",
            "Tracker entries exist",
            "Entries have correct format"
        ], "Skills Development Tracker database not found")

    # Step 2: Verify database schema
    schema_ok = False
    schema_err = ""
    if tracker_db_id:
        try:
            db_info = notion_utils.retrieve_database(notion, tracker_db_id)
            properties = db_info.get("properties", {})

            required_props = {
                "Name": "title",
                "Current Skill": "relation",
                "Current Proficiency": "rollup",
                "Target Proficiency": "number",
                "Gap": "formula",
                "Learning Resources": "rich_text",
                "Progress Notes": "rich_text",
            }

            for prop_name, expected_type in required_props.items():
                if prop_name not in properties:
                    schema_err = f"Property '{prop_name}' not found in database"
                    break
                if properties[prop_name]["type"] != expected_type:
                    found_type = properties[prop_name]['type']
                    schema_err = (f"Property '{prop_name}' has incorrect type. "
                                  f"Expected '{expected_type}', got '{found_type}'")
                    break
            else:
                # Verify Target Proficiency is percent format
                if (
                    properties["Target Proficiency"].get("number", {}).get("format")
                    != "percent"
                ):
                    schema_err = "Target Proficiency should have 'percent' format"
                else:
                    schema_ok = True

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            schema_err = f"Error retrieving database info: {e}"

    if tracker_db_id:
        sc.check("Database schema valid", schema_ok, schema_err)

    # Step 3: Get Skills database to check entries
    skills_db_id = notion_utils.find_database_in_block(notion, page_id, "Skills")
    if not sc.check("Skills database found", bool(skills_db_id),
                    "Skills database not found"):
        sc.fail_remaining([
            "Tracker entries exist",
            "Entries have correct format"
        ], "Skills database not found")

    # Get all skills with proficiency < 70%
    skills_below_70 = []
    if skills_db_id:
        try:
            skills_results = notion_utils.query_database(notion, skills_db_id).get(
                "results", []
            )
            for skill in skills_results:
                skill_level = (
                    skill.get("properties", {}).get("Skill Level", {}).get("number", 1.0)
                )
                if skill_level < 0.7:
                    skill_name = (
                        skill.get("properties", {}).get("Skill", {}).get("title", [])
                    )
                    if skill_name:
                        skill_name_text = skill_name[0].get("text", {}).get("content", "")
                        skills_below_70.append(
                            {
                                "name": skill_name_text,
                                "id": skill["id"],
                                "level": skill_level,
                            }
                        )
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            print(f"Error querying Skills database: {e}", file=sys.stderr)

    # Step 4: Verify entries in Skills Development Tracker
    entries_ok = False
    entries_err = ""
    entry_format_ok = True
    entry_format_err = ""

    if tracker_db_id and skills_db_id:
        try:
            tracker_results = notion_utils.query_database(notion, tracker_db_id).get(
                "results", []
            )

            if len(skills_below_70) > 0 and len(tracker_results) == 0:
                entries_err = "No entries found in Skills Development Tracker database"
            else:
                entries_ok = True

            # Verify each entry
            for entry in tracker_results:
                props = entry.get("properties", {})

                # Check name format
                name_prop = props.get("Name", {}).get("title", [])
                if not name_prop:
                    entry_format_ok = False
                    entry_format_err = "Entry missing Name property"
                    break
                name_text = name_prop[0].get("text", {}).get("content", "")
                if not name_text.endswith(" Development Plan"):
                    entry_format_ok = False
                    entry_format_err = (f"Entry name '{name_text}' doesn't follow "
                                        "expected format")
                    break

                # Check relation to Skills database
                skill_relation = props.get("Current Skill", {}).get("relation", [])
                if not skill_relation:
                    entry_format_ok = False
                    entry_format_err = f"Entry '{name_text}' missing Current Skill relation"
                    break

                # Check Target Proficiency (should be set)
                target_prof = props.get("Target Proficiency", {}).get("number")
                if target_prof is None:
                    entry_format_ok = False
                    entry_format_err = f"Entry '{name_text}' missing Target Proficiency"
                    break

                # Check Learning Resources
                learning_resources = props.get("Learning Resources", {}).get(
                    "rich_text", []
                )
                if not learning_resources:
                    entry_format_ok = False
                    entry_format_err = f"Entry '{name_text}' missing Learning Resources"
                    break

                # Check Progress Notes
                progress_notes = props.get("Progress Notes", {}).get("rich_text", [])
                if not progress_notes:
                    entry_format_ok = False
                    entry_format_err = f"Entry '{name_text}' missing Progress Notes"
                    break

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            entries_ok = False
            entries_err = f"Error querying Skills Development Tracker: {e}"
            entry_format_ok = False
            entry_format_err = f"Error querying Skills Development Tracker: {e}"

    if tracker_db_id and skills_db_id:
        sc.check("Tracker entries exist", entries_ok, entries_err)
        sc.check("Entries have correct format", entry_format_ok, entry_format_err)

    # Step 5: Verify callout block exists after Skills section
    all_blocks = notion_utils.get_all_blocks_recursively(notion, page_id)

    # Find Skills database block
    skills_db_block_index = None
    for i, block in enumerate(all_blocks):
        if (
            block.get("type") == "child_database"
            and block.get("child_database", {}).get("title") == "Skills"
        ):
            skills_db_block_index = i
            break

    if not sc.check("Skills DB block found", skills_db_block_index is not None,
                    "Could not find Skills database block"):
        sc.fail_remaining(["Callout found after Skills DB"], "Skills DB block not found")
        return sc.summary()

    # Look for callout block after Skills database
    callout_found = False
    callout_err = ""
    if skills_db_block_index + 1 < len(all_blocks):
        block = all_blocks[skills_db_block_index + 1]
        if block.get("type") == "callout":
            callout_data = block.get("callout", {})

            # Check background color
            if callout_data.get("color") != "blue_background":
                callout_err = "Could not find callout block with blue background"
            else:
                # Check icon
                icon = callout_data.get("icon", {})
                if icon.get("type") != "emoji" or icon.get("emoji") != "🎯":
                    callout_err = "Could not find callout block with 🎯 emoji"
                else:
                    # Check content starts with "Focus Areas:"
                    rich_text = callout_data.get("rich_text", [])
                    if rich_text:
                        content = rich_text[0].get("text", {}).get("content", "")
                        if (
                            content.startswith("Focus Areas:")
                            and "CSS + Basic JS" in content
                            and "Webflow" in content
                            and "Rive" in content
                        ):
                            callout_found = True
                            print(f"Success: Found callout block with content: {content}")
                        else:
                            callout_err = "Could not find callout block with required text content"
        else:
            callout_err = "Could not find callout block with Focus Areas after Skills section"
    else:
        callout_err = "No blocks after Skills database"

    sc.check("Callout found after Skills DB", callout_found, callout_err)

    if sc.summary()[0]:
        print(
            "Success: Skills Development Tracker database and callout block verified successfully."
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
