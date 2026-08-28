"""Verification module for Courses Internships Relation task in Notion workspace."""

# pylint: disable=duplicate-code,import-error,astroid-error

import sys
from typing import Optional
from notion_client import Client
from agentbrew.environments.notion.benchmark.evaluator.utils import notion_utils
from agentbrew.environments.notion.benchmark.evaluator.utils.notion_utils import SubCriteriaCollector

# ---------------------------------------------------------------------------
# Constants -----------------------------------------------------------------
# ---------------------------------------------------------------------------
MAIN_PAGE_TITLE = "Computer Science Student Dashboard"
COURSES_DB_TITLE = "Courses"
INTERNSHIP_DB_TITLE = "Internship search"

COURSE_CODES = {"CS301", "CS302", "CS303"}
COURSE_RELATION_NAME = "Related Internships"
INTERNSHIP_RELATION_NAME = "Relevant Courses"

INTERNSHIP_COMPANIES = {"OpenAI", "Google"}

# ---------------------------------------------------------------------------
# Helper functions -----------------------------------------------------------
# ---------------------------------------------------------------------------


def _locate_main_page(notion: Client, main_id: Optional[str]) -> Optional[str]:
    """Return the page_id of the dashboard page or None if not found."""
    page_id = None
    if main_id:
        found_id, obj_type = notion_utils.find_page_or_database_by_id(notion, main_id)
        if found_id and obj_type == "page":
            page_id = found_id
    if not page_id:
        page_id = notion_utils.find_page(notion, MAIN_PAGE_TITLE)
    return page_id


def _locate_database(notion: Client, parent_page_id: str, db_title: str) -> Optional[str]:
    """Recursively search for a child database by title and return its id."""
    return notion_utils.find_database_in_block(notion, parent_page_id, db_title)


# ---------------------------------------------------------------------------
# Verification logic ---------------------------------------------------------
# ---------------------------------------------------------------------------


def verify(notion: Client, main_id: Optional[str] = None) -> tuple[bool, str]:  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """Verify completion of the Courses ↔ Internship relation task."""
    sc = SubCriteriaCollector("Courses Internships Relation")

    # ------------------------------------------------------------------
    # Locate main page and databases -----------------------------------
    # ------------------------------------------------------------------
    page_id = _locate_main_page(notion, main_id)
    if not sc.check("Main page found", bool(page_id),
                    f"Page '{MAIN_PAGE_TITLE}' not found"):
        sc.fail_remaining([
            "Courses database found", "Internships database found",
            "Courses relation property exists", "Courses relation configured correctly",
            "Internships relation property exists", "Internships relation configured correctly",
            "Exactly 3 new course pages valid", "Exactly 2 new internship pages valid",
            "Course relations point to expected internships",
            "Internship relations point to expected courses"
        ], "main page not found")
        return sc.summary()

    courses_db_id = _locate_database(notion, page_id, COURSES_DB_TITLE)
    internships_db_id = _locate_database(notion, page_id, INTERNSHIP_DB_TITLE)

    if not sc.check("Courses database found", bool(courses_db_id),
                    f"Database '{COURSES_DB_TITLE}' not found"):
        sc.fail_remaining([
            "Courses relation property exists", "Courses relation configured correctly",
            "Exactly 3 new course pages valid",
            "Course relations point to expected internships"
        ], "Courses database not found")

    if not sc.check("Internships database found", bool(internships_db_id),
                    f"Database '{INTERNSHIP_DB_TITLE}' not found"):
        sc.fail_remaining([
            "Internships relation property exists", "Internships relation configured correctly",
            "Exactly 2 new internship pages valid",
            "Internship relations point to expected courses"
        ], "Internships database not found")

    if not courses_db_id or not internships_db_id:
        return sc.summary()

    # ------------------------------------------------------------------
    # Validate relation properties -------------------------------------
    # ------------------------------------------------------------------
    courses_db_obj = notion_utils.retrieve_database(notion, courses_db_id)
    internships_db_obj = notion_utils.retrieve_database(notion, internships_db_id)

    courses_props = courses_db_obj.get("properties", {})
    internships_props = internships_db_obj.get("properties", {})

    # Courses → Internships relation
    if not sc.check("Courses relation property exists",
                    COURSE_RELATION_NAME in courses_props,
                    f"Property '{COURSE_RELATION_NAME}' missing in Courses database"):
        sc.fail_remaining(["Courses relation configured correctly"],
                          "relation property missing")
    else:
        course_rel_prop = courses_props[COURSE_RELATION_NAME]
        sc.check("Courses relation configured correctly",
                 course_rel_prop.get("type") == "relation"
                 and course_rel_prop["relation"].get("database_id") == internships_db_id,
                 "Courses relation property is not configured correctly")

    # Internships → Courses relation
    if not sc.check("Internships relation property exists",
                    INTERNSHIP_RELATION_NAME in internships_props,
                    f"Property '{INTERNSHIP_RELATION_NAME}' missing in Internship search database"):
        sc.fail_remaining(["Internships relation configured correctly"],
                          "relation property missing")
    else:
        intern_rel_prop = internships_props[INTERNSHIP_RELATION_NAME]
        sc.check("Internships relation configured correctly",
                 intern_rel_prop.get("type") == "relation"
                 and intern_rel_prop["relation"].get("database_id") == courses_db_id,
                 "Internship relation property is not configured correctly")

    # ------------------------------------------------------------------
    # Validate course pages --------------------------------------------
    # ------------------------------------------------------------------
    course_pages = notion_utils.query_database(notion, courses_db_id).get("results", [])

    valid_course_count = 0
    course_page_id_set = set()
    internship_ids_seen: set[str] = set()
    courses_valid = True
    courses_err = ""

    for page in course_pages:
        props = page.get("properties", {})
        code_rts = props.get("Code", {}).get("rich_text", [])
        code_val = "".join(rt.get("plain_text", "") for rt in code_rts).strip()
        if code_val not in COURSE_CODES:
            continue  # not one of the new course entries we care about

        # Check required scalar props
        title_rts = props.get("Name", {}).get("title", [])
        name_ok = bool("".join(rt.get("plain_text", "") for rt in title_rts).strip())
        credits_ok = props.get("Credit", {}).get("number") is not None
        status_name = props.get("Status", {}).get("status", {}).get("name", "")
        status_allowed = {"planned", "in progress", "completed"}
        status_ok = status_name.lower() in status_allowed

        # Relation must point to at least one internship
        relations = props.get(COURSE_RELATION_NAME, {}).get("relation", [])
        if not (name_ok and credits_ok and status_ok and relations):
            msg = (f"Course '{code_val}' is missing required property "
                   "values or relations, or wrong values.")
            print(f"Error: {msg}", file=sys.stderr)
            courses_valid = False
            courses_err = msg
            continue

        # Collect IDs for further mutual check
        course_page_id_set.add(page["id"])
        internship_ids_seen.update(rel["id"] for rel in relations)
        valid_course_count += 1

    if valid_course_count != 3:
        sc.check("Exactly 3 new course pages valid", False,
                 f"Expected exactly 3 new course pages with codes {COURSE_CODES}, "
                 f"found {valid_course_count}")
    elif not courses_valid:
        sc.check("Exactly 3 new course pages valid", False, courses_err)
    else:
        sc.check("Exactly 3 new course pages valid", True)

    # ------------------------------------------------------------------
    # Validate internship pages ----------------------------------------
    # ------------------------------------------------------------------
    internship_pages = notion_utils.query_database(notion, internships_db_id).get(
        "results", []
    )

    valid_intern_count = 0
    internship_page_ids = set()
    course_ids_seen_from_intern: set[str] = set()
    interns_valid = True
    interns_err = ""

    for page in internship_pages:
        props = page.get("properties", {})
        company_rts = props.get("Company", {}).get("rich_text", [])
        company = "".join(rt.get("plain_text", "") for rt in company_rts).strip()
        if company not in INTERNSHIP_COMPANIES:
            continue  # not one of the two new internships

        role_rts = props.get("Role", {}).get("title", [])
        role_ok = bool("".join(rt.get("plain_text", "") for rt in role_rts).strip())
        status_name = props.get("Status", {}).get("status", {}).get("name", "")
        status_ok = status_name.lower() == "interested"
        relations = props.get(INTERNSHIP_RELATION_NAME, {}).get("relation", [])

        if not (role_ok and status_ok and relations):
            msg = (f"Internship at '{company}' is missing required "
                   "property values or relations, or wrong values.")
            print(f"Error: {msg}", file=sys.stderr)
            interns_valid = False
            interns_err = msg
            continue

        internship_page_ids.add(page["id"])
        course_ids_seen_from_intern.update(rel["id"] for rel in relations)
        valid_intern_count += 1

    if valid_intern_count != 2:
        sc.check("Exactly 2 new internship pages valid", False,
                 f"Expected exactly 2 new internship pages for companies "
                 f"{INTERNSHIP_COMPANIES}, found {valid_intern_count}")
    elif not interns_valid:
        sc.check("Exactly 2 new internship pages valid", False, interns_err)
    else:
        sc.check("Exactly 2 new internship pages valid", True)

    # ------------------------------------------------------------------
    # Mutual relation consistency --------------------------------------
    # ------------------------------------------------------------------
    sc.check("Course relations point to expected internships",
             internship_ids_seen.issubset(internship_page_ids),
             "Some course relations point to pages outside the expected internships")

    sc.check("Internship relations point to expected courses",
             course_ids_seen_from_intern.issubset(course_page_id_set),
             "Some internship relations point to pages outside the expected courses")

    return sc.summary()


# ---------------------------------------------------------------------------
# CLI entry-point -----------------------------------------------------------
# ---------------------------------------------------------------------------


def main() -> None:
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
