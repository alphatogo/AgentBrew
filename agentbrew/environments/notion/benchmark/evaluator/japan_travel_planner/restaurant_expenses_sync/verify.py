"""Verification module for Restaurant Expenses Sync task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def verify(notion: Client, main_id: str = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """
    Verifies that restaurants from Day 1 of Travel Itinerary have corresponding expense entries.
    """
    sc = SubCriteriaCollector("Restaurant Expenses Sync")

    page_id = None
    if main_id:
        found_id, object_type = notion_utils.find_page_or_database_by_id(
            notion, main_id
        )
        if found_id and object_type == "page":
            page_id = found_id

    if not page_id:
        page_id = notion_utils.find_page(notion, "Japan Travel Planner")

    if not sc.check("Japan Travel Planner page found", bool(page_id),
                    "Page 'Japan Travel Planner' not found"):
        sc.fail_remaining([
            "Travel Itinerary database found", "Expenses database found",
            "Day 1 restaurants found", "Restaurant names found",
            "All restaurants have matching expense entries"
        ], "main page not found")
        return sc.summary()

    # Find Travel Itinerary database
    itinerary_db_id = notion_utils.find_database_in_block(
        notion, page_id, "Travel Itinerary"
    )
    if not sc.check("Travel Itinerary database found", bool(itinerary_db_id),
                    "Database 'Travel Itinerary' not found"):
        sc.fail_remaining([
            "Day 1 restaurants found", "Restaurant names found",
            "All restaurants have matching expense entries"
        ], "Travel Itinerary database not found")

    # Find Expenses database
    expenses_db_id = notion_utils.find_database_in_block(notion, page_id, "Expenses")
    if not sc.check("Expenses database found", bool(expenses_db_id),
                    "Database 'Expenses' not found"):
        sc.fail_remaining([
            "All restaurants have matching expense entries"
        ], "Expenses database not found")

    if not itinerary_db_id or not expenses_db_id:
        return sc.summary()

    # Query Day 1 restaurants from Travel Itinerary
    itinerary_results = []
    itinerary_ok = False
    itinerary_err = ""
    try:
        itinerary_results = notion_utils.query_database(
            notion,
            itinerary_db_id,
            filter={
                "and": [
                    {"property": "Day", "select": {"equals": "Day 1"}},
                    {"property": "Type", "multi_select": {"contains": "Food"}},
                ]
            },
        ).get("results", [])
        itinerary_ok = True
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        itinerary_err = str(e)

    if not itinerary_ok:
        sc.check("Day 1 restaurants found", False,
                 f"Error querying Travel Itinerary database: {itinerary_err}")
        sc.fail_remaining(["Restaurant names found",
                           "All restaurants have matching expense entries"],
                          "itinerary query failed")
        return sc.summary()

    if not sc.check("Day 1 restaurants found", bool(itinerary_results),
                    "No restaurants found for Day 1 in Travel Itinerary"):
        sc.fail_remaining(["Restaurant names found",
                           "All restaurants have matching expense entries"],
                          "no Day 1 restaurants found")
        return sc.summary()

    # Extract restaurant names
    restaurant_names = []
    for entry in itinerary_results:
        props = entry.get("properties", {})
        name_prop = props.get("Name", {})
        name_text = "".join(t.get("plain_text", "") for t in name_prop.get("title", []))
        if name_text:
            restaurant_names.append(name_text.strip())

    if not sc.check("Restaurant names found", bool(restaurant_names),
                    "No restaurant names found in Day 1 entries"):
        sc.fail_remaining(["All restaurants have matching expense entries"],
                          "no restaurant names")
        return sc.summary()

    # Get descriptions from Japan Places to Visit database (same as Travel Itinerary)
    places_db_id = itinerary_db_id
    places_results = []
    try:
        places_results = notion_utils.query_database(notion, places_db_id).get(
            "results", []
        )
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"Error querying Japan Places to Visit database: {e}", file=sys.stderr)

    # Create a map of restaurant names to descriptions
    restaurant_descriptions = {}
    for place in places_results:
        props = place.get("properties", {})
        name_prop = props.get("Name", {})
        name_text = "".join(t.get("plain_text", "") for t in name_prop.get("title", []))

        desc_prop = props.get("Description", {})
        desc_text = "".join(
            t.get("plain_text", "") for t in desc_prop.get("rich_text", [])
        )

        if name_text and desc_text:
            restaurant_descriptions[name_text.strip()] = desc_text.strip()

    # Query Expenses database
    expenses_results = []
    expenses_ok = False
    expenses_err = ""
    try:
        expenses_results = notion_utils.query_database(notion, expenses_db_id).get(
            "results", []
        )
        expenses_ok = True
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        expenses_err = str(e)

    if not expenses_ok:
        sc.check("All restaurants have matching expense entries", False,
                 f"Error querying Expenses database: {expenses_err}")
        return sc.summary()

    # Verify each restaurant has a corresponding expense entry
    verified_restaurants = []
    all_matched = True
    unmatched_err = ""
    for restaurant_name in restaurant_names:
        found_matching_expense = False
        expected_description = restaurant_descriptions.get(restaurant_name, "")

        for expense in expenses_results:
            props = expense.get("properties", {})

            # Check Expense field (title)
            expense_prop = props.get("Expense", {})
            expense_text = "".join(
                t.get("plain_text", "") for t in expense_prop.get("title", [])
            )
            if expense_text.strip() != restaurant_name:
                continue

            # Check Date
            date_prop = props.get("Date", {})
            date_start = date_prop.get("date", {}).get("start")
            if date_start != "2025-01-01":
                continue

            # Check Transaction Amount
            amount_prop = props.get("Transaction Amount", {})
            amount = amount_prop.get("number")
            if amount != 120:
                continue

            # Check Category contains Dining
            category_prop = props.get("Category", {})
            categories = [c.get("name") for c in category_prop.get("multi_select", [])]
            if "Dining" not in categories:
                continue

            # Check Comment matches description (if description exists)
            if expected_description:
                comment_prop = props.get("Comment", {})
                comment_text = "".join(
                    t.get("plain_text", "") for t in comment_prop.get("rich_text", [])
                )
                if comment_text.strip().replace(
                    "\u202f", " "
                ) != expected_description.replace("\u202f", " "):
                    continue

            found_matching_expense = True
            verified_restaurants.append(restaurant_name)
            break

        if not found_matching_expense:
            print(
                f"Error: No matching expense entry found for restaurant '{restaurant_name}'.",
                file=sys.stderr,
            )
            all_matched = False
            unmatched_err = (f"No matching expense entry found for restaurant "
                             f"'{restaurant_name}'")

    if len(verified_restaurants) == len(restaurant_names):
        total_count = len(restaurant_names)
        msg = (f"Success: Found matching expense entries for all "
               f"{total_count} Day 1 restaurants.")
        print(msg)
        sc.check("All restaurants have matching expense entries", True)
    elif all_matched:
        verified_count = len(verified_restaurants)
        total_count = len(restaurant_names)
        sc.check("All restaurants have matching expense entries", False,
                 f"Only {verified_count} out of {total_count} restaurants "
                 "have matching expense entries")
    else:
        sc.check("All restaurants have matching expense entries", False, unmatched_err)

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
