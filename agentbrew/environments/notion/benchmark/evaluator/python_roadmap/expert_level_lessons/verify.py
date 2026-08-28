"""Verification module for Expert Level Lessons task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-return-statements,too-many-statements
    """
    Verifies that the Expert Level chapter and its lessons have been created
    correctly with complex prerequisites.
    """
    sc = SubCriteriaCollector("Expert Level Lessons")

    # Step 1: Find the main page and get database IDs
    found_id = None
    if main_id:
        _found_id, object_type = notion_utils.find_page_or_database_by_id(notion, main_id)
        if _found_id and object_type == 'page':
            found_id = _found_id

    if not found_id:
        found_id = notion_utils.find_page(notion, "Python Roadmap")

    if not sc.check("Python Roadmap page found", bool(found_id),
                    "Main page not found"):
        sc.fail_remaining([
            "Chapters database found", "Steps database found",
            "Expert Level chapter found", "Expert Level chapter icon correct",
            "Control Flow lesson has status Done",
            "Decorators lesson found", "Calling API lesson found",
            "Regular Expressions lesson found",
            "Advanced Foundations Review found",
            "Bridge lesson status Done", "Bridge linked to Expert Level",
            "Bridge parent is Control Flow", "Bridge has required subitems",
            "Expert lesson Metaprogramming found", "Expert lesson Async Concurrency found",
            "Expert lesson Memory Management found",
            "Expert lesson Building Python C Extensions found",
            "Building Python C Extensions parent correct",
            "Memory Management has 2 subitems",
            "Error Handling has Async Concurrency as subitem",
            "Bridge page content structure correct",
            "Done lessons count correct", "In Progress lessons count correct",
            "Expert Level has 5 lessons"
        ], "main page not found")
        return sc.summary()

    print(f"Found main page: {found_id}")

    all_blocks = notion_utils.get_all_blocks_recursively(notion, found_id)
    print(f"Found {len(all_blocks)} blocks")

    chapters_db_id = None
    chapters_db_title = None
    steps_db_id = None
    steps_db_title = None

    for block in all_blocks:
        if block and block.get("type") == "child_database":
            db_title = block.get("child_database", {}).get("title", "")
            if "Chapters" in db_title:
                chapters_db_id = block["id"]
                chapters_db_title = db_title
                print(f"Found Chapters database: {chapters_db_id}")
            elif "Steps" in db_title:
                steps_db_id = block["id"]
                steps_db_title = db_title
                print(f"Found Steps database: {steps_db_id}")

    if not sc.check("Chapters database found", bool(chapters_db_id),
                    "Chapters database not found"):
        sc.fail_remaining([
            "Expert Level chapter found", "Expert Level chapter icon correct",
            "Expert lesson Metaprogramming found", "Expert lesson Async Concurrency found",
            "Expert lesson Memory Management found",
            "Expert lesson Building Python C Extensions found",
            "Expert Level has 5 lessons"
        ], "Chapters database not found")

    if not sc.check("Steps database found", bool(steps_db_id),
                    "Steps database not found"):
        sc.fail_remaining([
            "Control Flow lesson has status Done",
            "Decorators lesson found", "Calling API lesson found",
            "Regular Expressions lesson found",
            "Advanced Foundations Review found",
            "Bridge lesson status Done", "Bridge linked to Expert Level",
            "Bridge parent is Control Flow", "Bridge has required subitems",
            "Expert lesson Metaprogramming found", "Expert lesson Async Concurrency found",
            "Expert lesson Memory Management found",
            "Expert lesson Building Python C Extensions found",
            "Building Python C Extensions parent correct",
            "Memory Management has 2 subitems",
            "Error Handling has Async Concurrency as subitem",
            "Bridge page content structure correct",
            "Done lessons count correct", "In Progress lessons count correct",
            "Expert Level has 5 lessons"
        ], "Steps database not found")
        return sc.summary()

    print("Starting verification...")

    def query_chapters(**kwargs):
        return notion_utils.query_database_with_fallback(
            notion, chapters_db_id, chapters_db_title, **kwargs
        )

    def query_steps(**kwargs):
        return notion_utils.query_database_with_fallback(
            notion, steps_db_id, steps_db_title, **kwargs
        )

    # Step 2: Verify the Expert Level chapter exists
    print("2. Checking for Expert Level chapter...")
    expert_chapter_id = None
    chapter_ok = False
    chapter_err = ""
    icon_ok = False
    icon_err = ""

    try:
        chapters_response = query_chapters(
            filter={
                "property": "Name",
                "title": {
                    "equals": "Expert Level"
                }
            }
        )

        if not chapters_response.get("results"):
            chapter_err = "Expert Level chapter not found in Chapters database"
        else:
            expert_chapter = chapters_response["results"][0]
            expert_chapter_id = expert_chapter["id"]
            chapter_ok = True

            # Check chapter icon (purple circle)
            chapter_icon = expert_chapter.get("icon")
            icon_type_ok = chapter_icon and chapter_icon.get("type") == "emoji"
            icon_emoji_ok = chapter_icon and chapter_icon.get("emoji") == "🟣"
            if not icon_type_ok or not icon_emoji_ok:
                icon_err = ("Expert Level chapter does not have the correct "
                            "purple circle emoji icon")
            else:
                icon_ok = True
                print("✓ Expert Level chapter found with correct icon: 🟣")

    except (ValueError, KeyError, TypeError, AttributeError) as e:
        chapter_err = f"Error querying Chapters database: {e}"
        icon_err = f"Error querying Chapters database: {e}"

    sc.check("Expert Level chapter found", chapter_ok, chapter_err)
    sc.check("Expert Level chapter icon correct", icon_ok, icon_err)

    # Step 3: Find Control Flow lesson (Done status)
    print("3. Finding Control Flow lesson...")
    control_flow_id = None
    control_flow_ok = False
    control_flow_err = ""

    try:
        control_flow_response = query_steps(
            filter={
                "and": [
                    {
                        "property": "Lessons",
                        "title": {
                            "contains": "Control"
                        }
                    },
                    {
                        "property": "Status",
                        "status": {
                            "equals": "Done"
                        }
                    }
                ]
            }
        )

        if control_flow_response.get("results"):
            control_flow_lesson = control_flow_response["results"][0]
            control_flow_id = control_flow_lesson["id"]
            control_flow_ok = True
            print("✓ Found Control Flow lesson with status 'Done'")
        else:
            control_flow_err = "Control Flow lesson not found with status 'Done'"

    except (ValueError, KeyError, TypeError, AttributeError) as e:
        control_flow_err = f"Error finding Control Flow lesson: {e}"

    sc.check("Control Flow lesson has status Done", control_flow_ok, control_flow_err)

    # Step 4: Find prerequisite lessons
    print("4. Finding prerequisite lessons...")
    decorators_id = None
    calling_api_id = None
    regex_id = None

    decorators_ok = False
    decorators_err = ""
    calling_api_ok = False
    calling_api_err = ""
    regex_ok = False
    regex_err = ""

    try:
        decorators_response = query_steps(
            filter={
                "property": "Lessons",
                "title": {
                    "contains": "Decorators"
                }
            }
        )

        if decorators_response.get("results"):
            decorators_lesson = decorators_response["results"][0]
            decorators_id = decorators_lesson["id"]
            if decorators_lesson["properties"]["Status"]["status"]["name"] != "Done":
                decorators_err = "Decorators lesson should have status 'Done'"
            else:
                decorators_ok = True
                print("✓ Found Decorators lesson with status 'Done'")
        else:
            decorators_err = "Decorators lesson not found"

        calling_api_response = query_steps(
            filter={
                "property": "Lessons",
                "title": {
                    "equals": "Calling API"
                }
            }
        )

        if calling_api_response.get("results"):
            calling_api_lesson = calling_api_response["results"][0]
            calling_api_id = calling_api_lesson["id"]
            calling_api_ok = True
            print("✓ Found Calling API lesson")
        else:
            calling_api_err = "Calling API lesson not found"

        regex_response = query_steps(
            filter={
                "property": "Lessons",
                "title": {
                    "contains": "Regular Expressions"
                }
            }
        )

        if regex_response.get("results"):
            regex_lesson = regex_response["results"][0]
            regex_id = regex_lesson["id"]
            regex_ok = True
            print("✓ Found Regular Expressions lesson")
        else:
            regex_err = "Regular Expressions lesson not found"

    except (ValueError, KeyError, TypeError, AttributeError) as e:
        decorators_err = decorators_err or f"Error finding prerequisite lessons: {e}"
        calling_api_err = calling_api_err or f"Error finding prerequisite lessons: {e}"
        regex_err = regex_err or f"Error finding prerequisite lessons: {e}"

    sc.check("Decorators lesson found", decorators_ok, decorators_err)
    sc.check("Calling API lesson found", calling_api_ok, calling_api_err)
    sc.check("Regular Expressions lesson found", regex_ok, regex_err)

    # Step 5: Verify Advanced Foundations Review bridge lesson
    print("5. Checking Advanced Foundations Review bridge lesson...")
    bridge_id = None
    bridge_found_ok = False
    bridge_found_err = ""
    bridge_status_ok = False
    bridge_status_err = ""
    bridge_chapter_ok = False
    bridge_chapter_err = ""
    bridge_parent_ok = False
    bridge_parent_err = ""
    bridge_subitems_ok = False
    bridge_subitems_err = ""

    try:
        bridge_response = query_steps(
            filter={
                "property": "Lessons",
                "title": {
                    "equals": "Advanced Foundations Review"
                }
            }
        )

        if not bridge_response.get("results"):
            bridge_found_err = "Advanced Foundations Review lesson not found"
            bridge_status_err = "lesson not found"
            bridge_chapter_err = "lesson not found"
            bridge_parent_err = "lesson not found"
            bridge_subitems_err = "lesson not found"
        else:
            bridge_lesson = bridge_response["results"][0]
            bridge_id = bridge_lesson["id"]
            bridge_found_ok = True

            # Check status is Done
            if bridge_lesson["properties"]["Status"]["status"]["name"] != "Done":
                bridge_status_err = "Advanced Foundations Review should have status 'Done'"
            else:
                bridge_status_ok = True

            # Check linked to Expert Level chapter
            if expert_chapter_id:
                bridge_chapters = bridge_lesson["properties"]["Chapters"]["relation"]
                if not any(rel["id"] == expert_chapter_id for rel in bridge_chapters):
                    bridge_chapter_err = ("Advanced Foundations Review not linked to "
                                          "Expert Level chapter")
                else:
                    bridge_chapter_ok = True
            else:
                bridge_chapter_err = "Expert Level chapter not found"

            # Check Parent item is Control Flow
            if control_flow_id:
                bridge_parent = bridge_lesson["properties"]["Parent item"]["relation"]
                if not bridge_parent or bridge_parent[0]["id"] != control_flow_id:
                    bridge_parent_err = ("Advanced Foundations Review should have Control "
                                         "Flow as Parent item")
                else:
                    bridge_parent_ok = True
            else:
                bridge_parent_err = "Control Flow lesson not found"

            # Check Sub-items
            if decorators_id and calling_api_id and regex_id:
                bridge_subitems = bridge_lesson["properties"]["Sub-item"]["relation"]
                required_subitems = {decorators_id, calling_api_id, regex_id}
                actual_subitems = {item["id"] for item in bridge_subitems}

                if not required_subitems.issubset(actual_subitems):
                    bridge_subitems_err = ("Advanced Foundations Review should have at least "
                                           "these 3 sub-items: Decorators, Calling API, "
                                           "Regular Expressions")
                elif len(bridge_subitems) < 5:
                    bridge_subitems_err = (f"Advanced Foundations Review should have at least "
                                           f"5 sub-items, found {len(bridge_subitems)}")
                else:
                    bridge_subitems_ok = True
                    subitems_count = len(bridge_subitems)
                    print(f"✓ Advanced Foundations Review has {subitems_count} sub-items")
            else:
                bridge_subitems_err = "Prerequisite lesson IDs not available"

    except (ValueError, KeyError, TypeError, AttributeError) as e:
        bridge_found_err = f"Error checking bridge lesson: {e}"
        bridge_status_err = f"Error: {e}"
        bridge_chapter_err = f"Error: {e}"
        bridge_parent_err = f"Error: {e}"
        bridge_subitems_err = f"Error: {e}"

    sc.check("Advanced Foundations Review found", bridge_found_ok, bridge_found_err)
    sc.check("Bridge lesson status Done", bridge_status_ok, bridge_status_err)
    sc.check("Bridge linked to Expert Level", bridge_chapter_ok, bridge_chapter_err)
    sc.check("Bridge parent is Control Flow", bridge_parent_ok, bridge_parent_err)
    sc.check("Bridge has required subitems", bridge_subitems_ok, bridge_subitems_err)

    # Step 6: Verify the 4 expert lessons
    print("6. Checking the 4 expert lessons...")

    error_handling_response = query_steps(
        filter={
            "property": "Lessons",
            "title": {
                "equals": "Error Handling"
            }
        }
    )

    error_handling_id = None
    if error_handling_response.get("results"):
        error_handling_id = error_handling_response["results"][0]["id"]
    else:
        sc.check("Error Handling lesson found", False, "Error Handling lesson not found")

    expert_lessons = {
        "Metaprogramming and AST Manipulation": {
            "status": "To Do",
            "parent": bridge_id,
            "date": "2025-09-15"
        },
        "Async Concurrency Patterns": {
            "status": "To Do",
            "parent": error_handling_id,
            "date": "2025-09-20"
        },
        "Memory Management and GC Tuning": {
            "status": "In Progress",
            "parent": bridge_id,
            "date": "2025-09-25"
        },
        "Building Python C Extensions": {
            "status": "To Do",
            "date": "2025-10-01"
        }
    }

    lesson_ids = {}

    try:
        for lesson_name, expected in expert_lessons.items():
            lesson_ok = True
            lesson_err = ""
            lesson_response = query_steps(
                filter={
                    "property": "Lessons",
                    "title": {
                        "equals": lesson_name
                    }
                }
            )

            if not lesson_response.get("results"):
                lesson_ok = False
                lesson_err = f"Lesson '{lesson_name}' not found"
            else:
                lesson = lesson_response["results"][0]
                lesson_ids[lesson_name] = lesson["id"]

                status_prop = lesson["properties"]["Status"]["status"]["name"]
                if status_prop != expected["status"]:
                    lesson_ok = False
                    lesson_err = (f"Lesson '{lesson_name}' should have status "
                                  f"'{expected['status']}'")

                lesson_chapters = lesson["properties"]["Chapters"]["relation"]
                if expert_chapter_id and not any(
                    rel["id"] == expert_chapter_id for rel in lesson_chapters
                ):
                    lesson_ok = False
                    lesson_err = f"Lesson '{lesson_name}' not linked to Expert Level chapter"

                lesson_date = lesson["properties"]["Date"]["date"]
                if lesson_date and lesson_date.get("start") != expected["date"]:
                    lesson_ok = False
                    lesson_err = (f"Lesson '{lesson_name}' should have date "
                                  f"'{expected['date']}'")

                if "parent" in expected and expected["parent"]:
                    lesson_parent = lesson["properties"]["Parent item"]["relation"]
                    if not lesson_parent or lesson_parent[0]["id"] != expected["parent"]:
                        lesson_ok = False
                        lesson_err = f"Lesson '{lesson_name}' should have correct parent item"

            safe_name = lesson_name.replace(" ", "_").replace("/", "_")[:30]
            sc.check(f"Expert lesson {safe_name}", lesson_ok, lesson_err)
            if lesson_ok:
                print(f"✓ Lesson '{lesson_name}' found with correct properties")

        # Special checks for Building Python C Extensions parent relationship
        building_c_ok = False
        building_c_err = ""
        metaprog_id = lesson_ids.get("Metaprogramming and AST Manipulation")
        if metaprog_id and "Building Python C Extensions" in lesson_ids:
            building_lesson = query_steps(
                filter={
                    "property": "Lessons",
                    "title": {
                        "equals": "Building Python C Extensions"
                    }
                }
            )["results"][0]

            building_parent = building_lesson["properties"]["Parent item"]["relation"]
            if not building_parent or building_parent[0]["id"] != metaprog_id:
                building_c_err = ("Building Python C Extensions should have "
                                  "Metaprogramming and AST Manipulation as parent")
            else:
                building_c_ok = True
        else:
            building_c_err = "Required lesson IDs not available"

        sc.check("Building Python C Extensions parent correct", building_c_ok, building_c_err)

        # Memory Management should have 2 sub-items
        memory_ok = False
        memory_err = ""
        if "Memory Management and GC Tuning" in lesson_ids:
            memory_lesson = query_steps(
                filter={
                    "property": "Lessons",
                    "title": {
                        "equals": "Memory Management and GC Tuning"
                    }
                }
            )["results"][0]

            memory_subitems = memory_lesson["properties"]["Sub-item"]["relation"]
            if len(memory_subitems) != 2:
                memory_err = ("Memory Management and GC Tuning should have "
                              "exactly 2 sub-items")
            else:
                memory_ok = True
        else:
            memory_err = "Memory Management lesson not found"

        sc.check("Memory Management has 2 subitems", memory_ok, memory_err)

    except (ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"Error checking expert lessons: {e}", file=sys.stderr)

    # Step 7: Verify Error Handling has Async Concurrency Patterns as sub-item
    print("7. Checking Error Handling sub-item...")
    error_subitem_ok = False
    error_subitem_err = ""

    try:
        error_handling_response2 = query_steps(
            filter={
                "property": "Lessons",
                "title": {
                    "equals": "Error Handling"
                }
            }
        )

        if error_handling_response2.get("results"):
            error_handling_lesson = error_handling_response2["results"][0]
            error_subitems = error_handling_lesson["properties"]["Sub-item"]["relation"]

            async_patterns_id = lesson_ids.get("Async Concurrency Patterns")
            if async_patterns_id and not any(
                item["id"] == async_patterns_id for item in error_subitems
            ):
                error_subitem_err = ("Error Handling should have Async Concurrency "
                                     "Patterns as sub-item")
            elif async_patterns_id:
                error_subitem_ok = True
                print("✓ Error Handling has Async Concurrency Patterns as sub-item")
            else:
                error_subitem_err = "Async Concurrency Patterns lesson ID not available"
        else:
            error_subitem_err = "Error Handling lesson not found"

    except (ValueError, KeyError, TypeError, AttributeError) as e:
        error_subitem_err = f"Error checking Error Handling: {e}"

    sc.check("Error Handling has Async Concurrency as subitem",
             error_subitem_ok, error_subitem_err)

    # Step 8: Verify block content in Advanced Foundations Review
    print("8. Checking Advanced Foundations Review page content...")
    bridge_content_ok = False
    bridge_content_err = ""

    if bridge_id:
        try:
            blocks = notion_utils.get_all_blocks_recursively(notion, bridge_id)

            if len(blocks) < 3:
                bridge_content_err = ("Advanced Foundations Review should have at least "
                                      "3 blocks")
            else:
                block1 = blocks[0]
                if block1.get("type") != "heading_2":
                    bridge_content_err = "First block should be heading_2"
                else:
                    heading_data = block1.get("heading_2", {}).get("rich_text", [{}])
                    heading_text = heading_data[0].get("text", {}).get("content", "")
                    if heading_text != "Prerequisites Checklist":
                        bridge_content_err = "Heading should be 'Prerequisites Checklist'"
                    else:
                        block2 = blocks[1]
                        if block2.get("type") != "bulleted_list_item":
                            bridge_content_err = "Second block should be bulleted_list_item"
                        elif len(blocks) >= 4:
                            block3 = blocks[2]
                            block4 = blocks[3]
                            if (block3.get("type") != "bulleted_list_item"
                                    or block4.get("type") != "bulleted_list_item"):
                                bridge_content_err = "Blocks 2-4 should be bulleted list items"
                            else:
                                last_block = blocks[-1]
                                if last_block.get("type") != "paragraph":
                                    bridge_content_err = "Last block should be paragraph"
                                else:
                                    para_data = last_block.get("paragraph", {}).get(
                                        "rich_text", [{}]
                                    )
                                    paragraph_text = para_data[0].get(
                                        "text", {}
                                    ).get("content", "")
                                    if "checkpoint" not in paragraph_text.lower():
                                        bridge_content_err = ("Paragraph should contain "
                                                               "text about checkpoint")
                                    else:
                                        bridge_content_ok = True
                                        print("✓ Advanced Foundations Review page "
                                              "has correct content structure")
                        else:
                            bridge_content_ok = True

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            bridge_content_err = f"Error checking page content: {e}"
    else:
        bridge_content_err = "Advanced Foundations Review not found"

    sc.check("Bridge page content structure correct", bridge_content_ok, bridge_content_err)

    # Step 9: Final verification counts
    print("9. Verifying final state counts...")
    done_count_ok = False
    done_count_err = ""
    in_progress_ok = False
    in_progress_err = ""
    expert_steps_ok = False
    expert_steps_err = ""

    try:
        all_lessons = query_steps(page_size=100)["results"]

        done_lessons = [
            l for l in all_lessons
            if l["properties"]["Status"]["status"]["name"] == "Done"
        ]
        done_count = len(done_lessons)
        in_progress_count = sum(
            1 for l in all_lessons
            if l["properties"]["Status"]["status"]["name"] == "In Progress"
        )

        if done_count != 14:
            print(f"Found {done_count} Done lessons (expected 14):", file=sys.stderr)
            for lesson in done_lessons:
                title_data = lesson["properties"]["Lessons"]["title"][0]
                lesson_name = title_data["text"]["content"]
                print(f"  - {lesson_name}", file=sys.stderr)
            done_count_err = f"Found {done_count} Done lessons (expected 14)"
        else:
            done_count_ok = True

        if in_progress_count != 1:
            in_progress_err = (f"Should have 1 In Progress lesson, "
                               f"found {in_progress_count}")
        else:
            in_progress_ok = True

        if chapters_db_id and expert_chapter_id:
            expert_chapter_updated = query_chapters(
                filter={
                    "property": "Name",
                    "title": {
                        "equals": "Expert Level"
                    }
                }
            )["results"][0]

            expert_steps = expert_chapter_updated["properties"]["Steps"]["relation"]
            if len(expert_steps) != 5:
                steps_count = len(expert_steps)
                expert_steps_err = (f"Expert Level should have exactly 5 lessons, "
                                    f"found {steps_count}")
            else:
                expert_steps_ok = True
                print("✓ Final state counts are correct")
        else:
            expert_steps_err = "Chapters database or Expert Level chapter not available"

    except (ValueError, KeyError, TypeError, AttributeError) as e:
        done_count_err = f"Error verifying final counts: {e}"
        in_progress_err = f"Error: {e}"
        expert_steps_err = f"Error: {e}"

    sc.check("Done lessons count correct", done_count_ok, done_count_err)
    sc.check("In Progress lessons count correct", in_progress_ok, in_progress_err)
    sc.check("Expert Level has 5 lessons", expert_steps_ok, expert_steps_err)

    if sc.summary()[0]:
        print("All verification checks passed!")
    return sc.summary()


def main():
    """
    Main verification function.
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
