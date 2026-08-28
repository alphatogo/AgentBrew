"""Verification module for Asset Retirement Migration task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
import re
from typing import Dict, Set, Optional
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector


def _get_database(root_page_id: str, notion: Client, name: str) -> Optional[str]:
    """Helper that finds a child database by title inside a page."""
    return notion_utils.find_database_in_block(notion, root_page_id, name)


def _check_property(props: Dict, name: str, expected_type: str) -> bool:
    if name not in props:
        print(f"Error: Property '{name}' missing in database.", file=sys.stderr)
        return False
    if props[name]["type"] != expected_type:
        found_type = props[name]['type']
        msg = (f"Error: Property '{name}' expected type '{expected_type}', "
               f"found '{found_type}'.")
        print(msg, file=sys.stderr)
        return False
    return True


def verify(notion: Client, main_id: Optional[str] = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """Verifies that the IT Asset Retirement Queue was created and populated correctly."""

    sc = SubCriteriaCollector("Asset Retirement Migration")

    # -------------------------------------------------------------------------
    # Resolve the root IT Trouble Shooting Hub page
    # -------------------------------------------------------------------------
    root_page_id = None
    if main_id:
        found_id, obj_type = notion_utils.find_page_or_database_by_id(notion, main_id)
        if found_id and obj_type == "page":
            root_page_id = found_id

    if not root_page_id:
        root_page_id = notion_utils.find_page(notion, "IT Trouble Shooting Hub")

    if not sc.check("IT Trouble Shooting Hub page found", bool(root_page_id),
                    "Could not locate the 'IT Trouble Shooting Hub' page"):
        sc.fail_remaining([
            "IT Inventory database found", "Retirement Queue database found",
            "Retirement Queue schema valid", "Retirement Reason options correct",
            "Retirement database description correct",
            "No expired items remaining in inventory",
            "Exactly 2 retirement pages", "Retirement pages have valid reasons",
            "Serial values match expected", "Migration log page found",
            "Migration log callout correct"
        ], "root page not found")
        return sc.summary()

    # -------------------------------------------------------------------------
    # Locate the original and new databases
    # -------------------------------------------------------------------------
    inventory_db_id = _get_database(root_page_id, notion, "IT Inventory")
    if not sc.check("IT Inventory database found", bool(inventory_db_id),
                    "'IT Inventory' database not found"):
        sc.fail_remaining([
            "No expired items remaining in inventory"
        ], "IT Inventory not found")

    retirement_db_id = _get_database(root_page_id, notion, "IT Asset Retirement Queue")
    if not sc.check("Retirement Queue database found", bool(retirement_db_id),
                    "'IT Asset Retirement Queue' database not found"):
        sc.fail_remaining([
            "Retirement Queue schema valid", "Retirement Reason options correct",
            "Retirement database description correct",
            "Exactly 2 retirement pages", "Retirement pages have valid reasons",
            "Serial values match expected"
        ], "Retirement Queue database not found")

    if retirement_db_id:
        # -------------------------------------------------------------------------
        # Validate schema of the retirement queue database
        # -------------------------------------------------------------------------
        retirement_db = notion_utils.retrieve_database(notion, retirement_db_id)
        r_props = retirement_db["properties"]

        required_schema = {
            "Serial": "title",
            "Tags": "multi_select",
            "Status": "select",
            "Vendor": "select",
            "Expiration date": "date",
            "Retirement Reason": "select",
        }

        schema_ok = True
        schema_err = ""
        for pname, ptype in required_schema.items():
            if not _check_property(r_props, pname, ptype):
                schema_ok = False
                schema_err = f"Property '{pname}' validation failed"
                break

        sc.check("Retirement Queue schema valid", schema_ok, schema_err)

        # Check Retirement Reason options
        expected_reason_options: Set[str] = {
            "Expired License",
            "Hardware Obsolete",
            "Security Risk",
            "User Offboarding",
        }
        actual_options = {
            opt["name"] for opt in r_props["Retirement Reason"]["select"]["options"]
        }
        if actual_options != expected_reason_options:
            print(
                "Error: 'Retirement Reason' select options mismatch.\n"
                f"Expected: {sorted(expected_reason_options)}\n"
                f"Found: {sorted(actual_options)}",
                file=sys.stderr,
            )
            expected_sorted = sorted(expected_reason_options)
            actual_sorted = sorted(actual_options)
            sc.check("Retirement Reason options correct", False,
                     f"'Retirement Reason' select options mismatch. "
                     f"Expected: {expected_sorted}, Found: {actual_sorted}")
        else:
            sc.check("Retirement Reason options correct", True)

        # ---------------------------------------------------------------
        # Validate database description starts with required phrase
        # ---------------------------------------------------------------
        desc_rich = retirement_db.get("description", [])
        desc_text = "".join([t.get("plain_text", "") for t in desc_rich])
        required_desc = "AUTO-GENERATED MIGRATION COMPLETED"
        sc.check("Retirement database description correct",
                 desc_text.strip() == required_desc,
                 f"Retirement database description must be exactly '{required_desc}'")

        # -------------------------------------------------------------------------
        # Validate that inventory items are moved & archived
        # -------------------------------------------------------------------------
        if inventory_db_id:
            expired_filter = {
                "property": "Status",
                "select": {"equals": "Expired"},
            }
            to_return_filter = {
                "property": "Status",
                "select": {"equals": "To be returned"},
            }
            compound_filter = {"or": [expired_filter, to_return_filter]}

            # Query for any *active* items that still match these statuses
            remaining_items = notion_utils.query_database(
                notion,
                inventory_db_id,
                filter=compound_filter,
                archived=False,
            ).get("results", [])

            sc.check("No expired items remaining in inventory",
                     not remaining_items,
                     f"{len(remaining_items)} 'Expired' / 'To be returned' items still present in IT Inventory")

        # There should be at least one entry in the retirement queue
        retirement_pages = notion_utils.query_database(notion, retirement_db_id).get(
            "results", []
        )
        expected_serials = {"65XYQ/GB", "36x10PIQ"}
        if len(retirement_pages) != len(expected_serials):
            sc.check("Exactly 2 retirement pages", False,
                     f"Expected {len(expected_serials)} retirement pages, "
                     f"found {len(retirement_pages)}")
            sc.fail_remaining(["Retirement pages have valid reasons", "Serial values match expected"],
                              "wrong number of retirement pages")
        else:
            sc.check("Exactly 2 retirement pages", True)

            # Each retirement page must have a Retirement Reason
            serials_seen = set()
            pages_valid = True
            pages_err = ""
            for page in retirement_pages:
                props = page["properties"]
                reason = props.get("Retirement Reason", {}).get("select", {})
                if not reason or reason.get("name") not in expected_reason_options:
                    print(f"Error: Page {page['id']} missing valid 'Retirement Reason'.",
                          file=sys.stderr)
                    pages_valid = False
                    pages_err = f"Page {page['id']} missing valid 'Retirement Reason'"

                # Collect Serial title
                title_rich = props.get("Serial", {}).get("title", [])
                serial_val = "".join([t.get("plain_text", "") for t in title_rich]).strip()
                serials_seen.add(serial_val)

            sc.check("Retirement pages have valid reasons", pages_valid, pages_err)

            sc.check("Serial values match expected",
                     serials_seen == expected_serials,
                     f"Serial values mismatch. Expected {sorted(expected_serials)}, "
                     f"found {sorted(serials_seen)}")

    # -----------------------------------------------------------------
    # Verify the migration log page and callout block contents
    # -----------------------------------------------------------------
    log_page_title = "Retirement Migration Log"
    log_page_id = notion_utils.find_page(notion, log_page_title)
    if not sc.check("Migration log page found", bool(log_page_id),
                    f"Page '{log_page_title}' not found"):
        sc.fail_remaining(["Migration log callout correct"], "log page not found")
        return sc.summary()

    # Search for a callout block with required pattern
    callout_pattern = re.compile(
        r"Successfully migrated (\d+) assets to the retirement queue "
        r"on 2025-03-24\."
    )
    blocks = notion_utils.get_all_blocks_recursively(notion, log_page_id)
    match_found = False
    callout_err = "Required callout block not found in migration log page"
    expected_serials = {"65XYQ/GB", "36x10PIQ"}  # re-define for use here
    for blk in blocks:
        if blk.get("type") == "callout":
            text = notion_utils.get_block_plain_text(blk)
            m = callout_pattern.search(text)
            if m:
                migrated_num = int(m.group(1))
                if migrated_num == len(expected_serials):
                    match_found = True
                else:
                    callout_err = (f"Callout reports {migrated_num} assets, "
                                   f"but {len(expected_serials)} retirement pages found")
                break

    sc.check("Migration log callout correct", match_found, callout_err)

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
