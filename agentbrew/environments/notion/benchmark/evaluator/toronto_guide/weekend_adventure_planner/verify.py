#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Verification module for Weekend Adventure Planner task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-return-statements,too-many-statements
    """
    Verifies that the Perfect Weekend Adventure page has been created correctly.
    """
    sc = SubCriteriaCollector("Weekend Adventure Planner")

    # Find the main Toronto Guide page
    page_id = None
    if main_id:
        found_id, object_type = notion_utils.find_page_or_database_by_id(notion, main_id)
        if found_id and object_type == "page":
            page_id = found_id

    if not page_id:
        page_id = notion_utils.find_page(notion, "Toronto Guide")
    if not sc.check("Toronto Guide page exists", bool(page_id), "Main 'Toronto Guide' page not found."):
        sc.fail_remaining(
            [
                "Perfect Weekend Adventure page exists",
                "Activities database query succeeds",
                "Food database query succeeds",
                "Cafes database query succeeds",
                "All required headings exist",
                "Beach activities list exists",
                "Beach activities count matches source data",
                "Cultural dining list exists",
                "Cultural dining count matches source data",
                "Top Cafes toggle exists",
                "Cafe to-do count matches source data",
                "All cafe to-dos are unchecked",
                "Weekend summary matches source counts",
                "Divider exists",
                "Pro tip callout exists",
            ],
            "main page not found",
        )
        return sc.summary()

    # Find the Perfect Weekend Adventure child page
    adventure_page_id = None
    search_error = ""
    try:
        response = notion.search(
            query="Perfect Weekend Adventure",
            filter={"property": "object", "value": "page"}
        )

        for result in response.get("results", []):
            parent = result.get("parent", {})
            if parent.get("type") == "page_id" and parent.get("page_id") == page_id:
                adventure_page_id = result["id"]
                break

        if not adventure_page_id:
            for result in response.get("results", []):
                title_list = result.get("properties", {}).get("title", {}).get("title", [])
                for title_obj in title_list:
                    if "Perfect Weekend Adventure" in title_obj.get("plain_text", ""):
                        adventure_page_id = result["id"]
                        break
                if adventure_page_id:
                    break
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        search_error = str(e)

    if not sc.check(
        "Perfect Weekend Adventure page exists",
        bool(adventure_page_id) and not search_error,
        search_error or "'Perfect Weekend Adventure' page not found as child of main page.",
    ):
        sc.fail_remaining(
            [
                "Activities database query succeeds",
                "Food database query succeeds",
                "Cafes database query succeeds",
                "All required headings exist",
                "Beach activities list exists",
                "Beach activities count matches source data",
                "Cultural dining list exists",
                "Cultural dining count matches source data",
                "Top Cafes toggle exists",
                "Cafe to-do count matches source data",
                "All cafe to-dos are unchecked",
                "Weekend summary matches source counts",
                "Divider exists",
                "Pro tip callout exists",
            ],
            "adventure page not found",
        )
        return sc.summary()

    # Get all blocks from the adventure page
    all_blocks = notion_utils.get_all_blocks_recursively(notion, adventure_page_id)

    # Get databases from the main Toronto Guide page
    activities_db_id = None
    food_db_id = None
    cafes_db_id = None
    activities_db_title = None
    food_db_title = None
    cafes_db_title = None

    main_blocks = notion_utils.get_all_blocks_recursively(notion, page_id)
    for block in main_blocks:
        if block.get("type") == "child_database":
            title = block.get("child_database", {}).get("title", "")
            if "Activities" in title:
                activities_db_id = block.get("id")
                activities_db_title = title
            elif "Food" in title:
                food_db_id = block.get("id")
                food_db_title = title
            elif "Cafes" in title or "Caf�" in title:
                cafes_db_id = block.get("id")
                cafes_db_title = title

    # Query databases to get expected data
    beach_activities = []
    cultural_restaurants = []
    cafes_list = []

    activities_error = ""
    if activities_db_id:  # pylint: disable=too-many-nested-blocks
        try:
            db_response = notion_utils.query_database_with_fallback(
                notion, activities_db_id, activities_db_title
            )
            for page in db_response.get("results", []):
                properties = page.get("properties", {})
                tags_prop = properties.get("Tags", {})
                if tags_prop.get("type") == "multi_select":
                    tags = [tag.get("name") for tag in tags_prop.get("multi_select", [])]
                    if "Beaches" in tags:
                        name_prop = properties.get("Name", {})
                        if name_prop.get("type") == "title" and name_prop.get("title"):
                            name = name_prop["title"][0]["plain_text"]
                            url_prop = properties.get("Google Maps Link", {})
                            url = url_prop.get("url", "") if url_prop.get("type") == "url" else ""
                            beach_activities.append({"name": name, "url": url})
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            activities_error = str(e)
    else:
        activities_error = "Activities database not found"

    food_error = ""
    if food_db_id:  # pylint: disable=too-many-nested-blocks
        try:
            db_response = notion_utils.query_database_with_fallback(
                notion, food_db_id, food_db_title
            )
            for page in db_response.get("results", []):
                properties = page.get("properties", {})
                tags_prop = properties.get("Tags", {})
                if tags_prop.get("type") == "multi_select":
                    tags = [tag.get("name") for tag in tags_prop.get("multi_select", [])]
                    for tag in tags:
                        if tag in ["Turkish", "Hakka"]:
                            name_prop = properties.get("Name", {})
                            if name_prop.get("type") == "title" and name_prop.get("title"):
                                name = name_prop["title"][0]["plain_text"]
                                cultural_restaurants.append({"name": name, "tag": tag})
                                break
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            food_error = str(e)
    else:
        food_error = "Food database not found"

    cafes_error = ""
    if cafes_db_id:
        try:
            db_response = notion_utils.query_database_with_fallback(
                notion, cafes_db_id, cafes_db_title
            )
            for page in db_response.get("results", []):
                properties = page.get("properties", {})
                name_prop = properties.get("Name", {})
                if name_prop.get("type") == "title" and name_prop.get("title"):
                    name = name_prop["title"][0]["plain_text"]
                    cafes_list.append(name)
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            cafes_error = str(e)
    else:
        cafes_error = "Cafes database not found"

    sc.check("Activities database query succeeds", not activities_error, activities_error)
    sc.check("Food database query succeeds", not food_error, food_error)
    sc.check("Cafes database query succeeds", not cafes_error, cafes_error)

    # Required headings and their types
    required_headings = [
        ("🎒 Perfect Weekend Adventure", "heading_1"),
        ("🏖️ Beach Activities", "heading_2"),
        ("🍽️ Cultural Dining Experience", "heading_2"),
        ("☕ Coffee Break Spots", "heading_2"),
        ("📊 Weekend Summary", "heading_2")
    ]

    # Track verification results
    found_headings = set()
    found_beach_list = False
    found_restaurant_list = False
    found_toggle_with_cafes = False
    found_summary = False
    found_divider = False
    found_callout = False

    # Variables to track counts
    beach_count = 0
    restaurant_count = 0
    cafe_count = 0

    current_section = None
    is_in_toggle = False
    checked_cafes = []

    for block in all_blocks:
        block_type = block.get("type")
        block_text = notion_utils.get_block_plain_text(block)

        # Check headings
        for heading_text, expected_type in required_headings:
            if heading_text in block_text and block_type == expected_type:
                found_headings.add(heading_text)
                current_section = heading_text

        # Check Beach Activities section
        if current_section == "🏖️ Beach Activities" and block_type == "bulleted_list_item":
            found_beach_list = True
            beach_count += 1
            # Verify format includes name and potentially URL
            for activity in beach_activities:
                if activity["name"] in block_text:
                    if activity["url"] and activity["url"] not in block_text:
                        msg = f"Warning: Beach activity '{activity['name']}' missing URL"
                        print(msg, file=sys.stderr)

        # Check Cultural Dining section
        elif (current_section == "🍽️ Cultural Dining Experience" and
              block_type == "numbered_list_item"):
            found_restaurant_list = True
            restaurant_count += 1
            # Check format: Restaurant Name (Tag: [tag])
            for restaurant in cultural_restaurants:
                tag_text = f"Tag: {restaurant['tag']}"
                if restaurant["name"] in block_text and tag_text in block_text:
                    pass  # Format is correct

        # Check Coffee Break Spots section
        elif current_section == "☕ Coffee Break Spots":
            if block_type == "toggle" and "Top Cafes to Visit" in block_text:
                is_in_toggle = True
                found_toggle_with_cafes = True
            elif is_in_toggle and block_type == "to_do":
                cafe_count += 1
                # Verify unchecked status
                to_do_data = block.get("to_do", {})
                if to_do_data.get("checked", False):
                    checked_cafes.append(block_text)
            elif block_type in ["heading_1", "heading_2", "heading_3"]:
                is_in_toggle = False

        # Check Weekend Summary section
        elif current_section == "📊 Weekend Summary" and block_type == "paragraph":
            beach_count_val = len(beach_activities)
            restaurant_count_val = len(cultural_restaurants)
            cafe_count_val = len(cafes_list)
            expected_text = (f"This weekend includes {beach_count_val} beach "
                             f"activities, {restaurant_count_val} cultural "
                             f"dining options, and {cafe_count_val} coffee "
                             f"spots to explore!")
            if expected_text in block_text:
                found_summary = True

        # Check for divider after summary
        if block_type == "divider":
            found_divider = True

        # Check for callout with pro tip
        if block_type == "callout":
            callout_data = block.get("callout", {})
            icon = callout_data.get("icon", {})
            if icon.get("type") == "emoji" and icon.get("emoji") == "💡":
                pro_tip_text = ("Pro tip: Check the Seasons database for the "
                                "best time to enjoy outdoor activities!")
                if pro_tip_text in block_text:
                    found_callout = True

    # Verify all required elements
    all_passed = True

    missing_headings = [heading_text for heading_text, _ in required_headings if heading_text not in found_headings]
    sc.check("All required headings exist", not missing_headings, f"Missing headings: {missing_headings}")
    sc.check("Beach activities list exists", found_beach_list, "Beach activities bulleted list not found")
    sc.check(
        "Beach activities count matches source data",
        beach_count == len(beach_activities),
        f"Expected {len(beach_activities)} beach activities, found {beach_count}",
    )
    sc.check("Cultural dining list exists", found_restaurant_list, "Cultural dining numbered list not found")
    sc.check(
        "Cultural dining count matches source data",
        restaurant_count == len(cultural_restaurants),
        f"Expected {len(cultural_restaurants)} cultural restaurants, found {restaurant_count}",
    )
    sc.check("Top Cafes toggle exists", found_toggle_with_cafes, "Toggle block 'Top Cafes to Visit' not found")
    sc.check(
        "Cafe to-do count matches source data",
        cafe_count == len(cafes_list),
        f"Expected {len(cafes_list)} cafes, found {cafe_count}",
    )
    sc.check(
        "All cafe to-dos are unchecked",
        not checked_cafes,
        f"Cafe to-do items should be unchecked: {checked_cafes}",
    )
    sc.check("Weekend summary matches source counts", found_summary, "Weekend summary with correct counts not found")
    sc.check("Divider exists", found_divider, "Divider block not found after summary")
    sc.check("Pro tip callout exists", found_callout, "Callout with pro tip not found")

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
