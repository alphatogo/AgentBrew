"""Deterministic Notion template context collection for task generation."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

import requests


NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"

KEEP_BLOCK_TYPES = {
    "heading_1",
    "heading_2",
    "heading_3",
    "paragraph",
    "bulleted_list_item",
    "numbered_list_item",
    "to_do",
    "toggle",
    "callout",
    "divider",
    "code",
    "quote",
    "column_list",
    "column",
    "child_page",
    "child_database",
}


class NotionAPIError(RuntimeError):
    """Raised when Notion REST API calls fail."""


def rich_text_to_plain(rich_text: list[dict[str, Any]]) -> str:
    return "".join(item.get("plain_text", "") for item in rich_text or [])


def get_page_title(page_obj: dict[str, Any]) -> str:
    for prop in (page_obj.get("properties") or {}).values():
        if prop.get("type") == "title":
            return rich_text_to_plain(prop.get("title") or [])
    return "Untitled"


def simplify_property_value(prop: dict[str, Any]) -> Any:
    ptype = prop.get("type")
    value = prop.get(ptype)
    if ptype in {"title", "rich_text"}:
        return rich_text_to_plain(value or [])
    if ptype in {"number", "checkbox", "url", "email", "phone_number", "date"}:
        return value
    if ptype in {"status", "select"}:
        return value.get("name") if value else None
    if ptype == "multi_select":
        return [item.get("name") for item in value or []]
    if ptype == "relation":
        return [item.get("id") for item in value or []]
    if ptype in {"formula", "rollup"} and isinstance(value, dict):
        return value.get(value.get("type"))
    return None


def extract_block_text(block: dict[str, Any]) -> str:
    btype = block.get("type")
    payload = block.get(btype) or {}
    if not isinstance(payload, dict):
        return ""
    if "rich_text" in payload:
        return rich_text_to_plain(payload.get("rich_text") or [])
    if btype in {"child_page", "child_database"}:
        return payload.get("title", "")
    if "caption" in payload:
        return rich_text_to_plain(payload.get("caption") or [])
    return ""


def normalize_block_min(block: dict[str, Any]) -> dict[str, Any]:
    btype = block.get("type")
    node: dict[str, Any] = {
        "id": block.get("id"),
        "type": btype,
        "text": extract_block_text(block),
        "children": [],
    }
    if btype == "to_do":
        node["checked"] = (block.get("to_do") or {}).get("checked", False)
    if btype == "callout" and (block.get("callout") or {}).get("icon"):
        node["icon"] = (block.get("callout") or {}).get("icon")
    if btype == "code":
        node["language"] = (block.get("code") or {}).get("language")
    return node


def normalize_schema(ds_obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, prop in (ds_obj.get("properties") or {}).items():
        ptype = prop.get("type")
        entry: dict[str, Any] = {"type": ptype}
        if ptype in {"select", "multi_select", "status"}:
            entry["options"] = [
                item.get("name")
                for item in ((prop.get(ptype) or {}).get("options") or [])
            ]
        elif ptype == "relation":
            relation = prop.get("relation") or {}
            entry["data_source_id"] = relation.get("data_source_id")
            entry["dual_property"] = relation.get("dual_property")
            entry["dual_data_source"] = relation.get("dual_data_source")
        elif ptype == "rollup":
            rollup = prop.get("rollup") or {}
            entry["relation_property_name"] = rollup.get("relation_property_name")
            entry["rollup_property_name"] = rollup.get("rollup_property_name")
            entry["function"] = rollup.get("function")
        elif ptype == "formula":
            entry["expression"] = (prop.get("formula") or {}).get("expression")
        out[name] = entry
    return out


def row_title(row: dict[str, Any]) -> str:
    for prop in (row.get("properties") or {}).values():
        if prop.get("type") == "title":
            return rich_text_to_plain(prop.get("title") or [])
    return "Untitled"


def truncate_text(text: Any, max_len: int) -> Any:
    if not isinstance(text, str) or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def compact_examples(values: list[Any], max_values: int, max_len: int) -> list[Any]:
    compacted: list[Any] = []
    for value in values[:max_values]:
        if isinstance(value, list) and value:
            item = truncate_text(value[0], max_len)
            count = value[1] if len(value) > 1 else None
            compacted.append([item, count] if count is not None else item)
        else:
            compacted.append(truncate_text(value, max_len))
    if len(values) > max_values:
        compacted.append(f"[omitted {len(values) - max_values} similar values]")
    return compacted


class NotionRestClient:
    """Small REST wrapper matching the old MCP-Universe data sampling code."""

    def __init__(self, api_key: str, timeout: int = 30, max_retries: int = 5) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            }
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{NOTION_API_BASE}{path}"
        for attempt in range(1, self.max_retries + 1):
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                timeout=self.timeout,
            )
            if response.status_code == 429:
                time.sleep(int(response.headers.get("Retry-After", "2")))
                continue
            if 500 <= response.status_code < 600 and attempt < self.max_retries:
                time.sleep(min(2**attempt, 10))
                continue
            if not response.ok:
                try:
                    payload = response.json()
                except Exception:  # pylint: disable=broad-exception-caught
                    payload = {"raw_text": response.text}
                raise NotionAPIError(f"{response.status_code} {method} {path}: {payload}")
            return response.json()
        raise NotionAPIError(f"Failed after retries: {method} {path}")

    def paginate(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        next_cursor = None
        while True:
            page_params = dict(params or {})
            page_body = dict(json_body or {})
            if method.upper() == "GET":
                page_params["page_size"] = 100
                if next_cursor:
                    page_params["start_cursor"] = next_cursor
            else:
                page_body["page_size"] = 100
                if next_cursor:
                    page_body["start_cursor"] = next_cursor
            payload = self.request(
                method,
                path,
                params=page_params if method.upper() == "GET" else None,
                json_body=page_body if method.upper() != "GET" else None,
            )
            out.extend(payload.get("results") or [])
            if not payload.get("has_more"):
                return out
            next_cursor = payload.get("next_cursor")

    def retrieve_page(self, page_id: str) -> dict[str, Any]:
        return self.request("GET", f"/pages/{page_id}")

    def retrieve_block_children(self, block_id: str) -> list[dict[str, Any]]:
        return self.paginate("GET", f"/blocks/{block_id}/children")

    def retrieve_database(self, database_id: str) -> dict[str, Any]:
        return self.request("GET", f"/databases/{database_id}")

    def retrieve_data_source(self, data_source_id: str) -> dict[str, Any]:
        return self.request("GET", f"/data_sources/{data_source_id}")

    def query_data_source(self, data_source_id: str) -> list[dict[str, Any]]:
        return self.paginate("POST", f"/data_sources/{data_source_id}/query", json_body={})


class BenchmarkSampler:
    """Collect page, block, database, and data-source facts from one template."""

    def __init__(self, client: NotionRestClient, max_rows_per_ds: int = 20, max_pages: int = 500) -> None:
        self.client = client
        self.max_rows_per_ds = max_rows_per_ds
        self.max_pages = max_pages
        self.pages: dict[str, dict[str, Any]] = {}
        self.data_sources: dict[str, dict[str, Any]] = {}
        self.visited_pages: set[str] = set()
        self.visited_blocks: set[str] = set()
        self.visited_databases: set[str] = set()
        self.visited_data_sources: set[str] = set()

    def run(self, root_page_id: str) -> dict[str, Any]:
        self._crawl_page(root_page_id)
        root_page = self.pages[root_page_id]
        return {
            "root": {"id": root_page["id"], "title": root_page["title"], "url": root_page["url"]},
            "pages": self.pages,
            "data_sources": self.data_sources,
            "facts": self._build_facts(),
        }

    def _crawl_page(self, page_id: str) -> None:
        if page_id in self.visited_pages or len(self.visited_pages) >= self.max_pages:
            return
        self.visited_pages.add(page_id)
        page_obj = self.client.retrieve_page(page_id)
        page_entry = {
            "id": page_id,
            "title": get_page_title(page_obj),
            "url": page_obj.get("url"),
            "children_pages": [],
            "databases": [],
            "outline": [],
        }
        self.pages[page_id] = page_entry
        page_entry["outline"] = self._crawl_blocks(page_id, page_entry)
        for child_page_id in list(page_entry["children_pages"]):
            self._crawl_page(child_page_id)

    def _crawl_blocks(self, block_id: str, page_entry: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            children = self.client.retrieve_block_children(block_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return []
        out: list[dict[str, Any]] = []
        for block in children:
            child_id = block.get("id")
            btype = block.get("type")
            if btype == "child_database":
                self._collect_database(child_id)
            if btype not in KEEP_BLOCK_TYPES:
                if block.get("has_children") and child_id not in self.visited_blocks:
                    self.visited_blocks.add(child_id)
                    out.extend(self._crawl_blocks(child_id, page_entry))
                continue
            node = normalize_block_min(block)
            out.append(node)
            if btype == "child_page" and child_id not in page_entry["children_pages"]:
                page_entry["children_pages"].append(child_id)
            if btype == "child_database" and child_id not in page_entry["databases"]:
                page_entry["databases"].append(child_id)
            if block.get("has_children") and child_id not in self.visited_blocks:
                self.visited_blocks.add(child_id)
                node["children"] = self._crawl_blocks(child_id, page_entry)
        return out

    def _collect_database(self, database_id: str) -> None:
        if not database_id or database_id in self.visited_databases:
            return
        self.visited_databases.add(database_id)
        try:
            db_obj = self.client.retrieve_database(database_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return
        for ds in db_obj.get("data_sources") or []:
            if ds.get("id"):
                self._collect_data_source(ds["id"])

    def _collect_data_source(self, data_source_id: str) -> None:
        if data_source_id in self.visited_data_sources:
            return
        self.visited_data_sources.add(data_source_id)
        try:
            ds_obj = self.client.retrieve_data_source(data_source_id)
            rows = self.client.query_data_source(data_source_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return
        schema = normalize_schema(ds_obj)
        sample_rows = []
        counters: dict[str, Counter] = defaultdict(Counter)
        for row in rows[: self.max_rows_per_ds]:
            props = {
                name: simplify_property_value(prop)
                for name, prop in (row.get("properties") or {}).items()
            }
            sample_rows.append({"id": row.get("id"), "title": row_title(row), "properties": props})
        for row in rows:
            for name, prop in (row.get("properties") or {}).items():
                value = simplify_property_value(prop)
                if isinstance(value, str) and value:
                    counters[name][value] += 1
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item:
                            counters[name][item] += 1
        self.data_sources[data_source_id] = {
            "id": data_source_id,
            "name": ds_obj.get("name"),
            "schema": schema,
            "sample_rows": sample_rows,
            "stats": {
                "row_count": len(rows),
                "top_values": {
                    name: counter.most_common(10)
                    for name, counter in counters.items()
                    if counter
                },
            },
        }

    def _build_facts(self) -> dict[str, Any]:
        page_facts = []
        for page in self.pages.values():
            headings: list[dict[str, str]] = []
            flags = {"has_toggle": False, "has_columns": False, "has_callout": False}

            def walk(nodes: list[dict[str, Any]]) -> None:
                for node in nodes:
                    if node["type"] in {"heading_1", "heading_2", "heading_3"} and node["text"]:
                        headings.append({"type": node["type"], "text": node["text"]})
                    flags["has_toggle"] = flags["has_toggle"] or node["type"] == "toggle"
                    flags["has_columns"] = flags["has_columns"] or node["type"] in {"column_list", "column"}
                    flags["has_callout"] = flags["has_callout"] or node["type"] == "callout"
                    walk(node.get("children") or [])

            walk(page["outline"])
            page_facts.append(
                {
                    "page_id": page["id"],
                    "title": page["title"],
                    "headings": headings,
                    **flags,
                    "database_count": len(page["databases"]),
                    "child_page_count": len(page["children_pages"]),
                }
            )
        ds_facts = []
        for ds in self.data_sources.values():
            schema = ds["schema"]
            ds_facts.append(
                {
                    "data_source_id": ds["id"],
                    "name": ds["name"],
                    "property_names": list(schema.keys()),
                    "property_types": {key: value["type"] for key, value in schema.items()},
                    "has_relation": any(value["type"] == "relation" for value in schema.values()),
                    "has_rollup": any(value["type"] == "rollup" for value in schema.values()),
                    "has_formula": any(value["type"] == "formula" for value in schema.values()),
                    "row_count": ds["stats"]["row_count"],
                }
            )
        return {"pages": page_facts, "data_sources": ds_facts}


def infer_task_types(template_result: dict[str, Any]) -> list[str]:
    facts = template_result.get("facts") or {}
    page_facts = facts.get("pages") or []
    ds_facts = facts.get("data_sources") or []
    task_types = set()
    if any(page.get("child_page_count", 0) > 0 for page in page_facts):
        task_types.add("create_child_page")
    if ds_facts or any(page.get("database_count", 0) > 0 for page in page_facts):
        task_types.add("create_or_update_database")
    if any(ds.get("row_count", 0) > 0 for ds in ds_facts):
        task_types.update({"populate_database_rows", "summarize_existing_data"})
    if any(page.get("has_toggle") or page.get("has_columns") for page in page_facts):
        task_types.add("reorganize_page_layout")
    if any(page.get("has_callout") for page in page_facts):
        task_types.add("insert_or_update_callout")
    if any(ds.get("has_relation") for ds in ds_facts):
        task_types.add("build_relations")
    if any(ds.get("has_rollup") or ds.get("has_formula") for ds in ds_facts):
        task_types.add("advanced_database_modeling")
    if not task_types and page_facts:
        task_types.add("simple_page_edit")
    return sorted(task_types)


def assess_benchmark_support(template_result: dict[str, Any]) -> dict[str, Any]:
    facts = template_result.get("facts") or {}
    page_facts = facts.get("pages") or []
    ds_facts = facts.get("data_sources") or []
    signals = {
        "has_multiple_pages": len(page_facts) >= 2,
        "has_headings": any(page.get("headings") for page in page_facts),
        "has_data_source": len(ds_facts) >= 1,
        "has_nontrivial_data_source": any(ds.get("row_count", 0) >= 3 for ds in ds_facts),
        "has_schema_constraints": any(len(ds.get("property_names") or []) >= 3 for ds in ds_facts),
        "has_layout_blocks": any(
            page.get("has_toggle") or page.get("has_columns") or page.get("has_callout")
            for page in page_facts
        ),
        "has_relational_features": any(
            ds.get("has_relation") or ds.get("has_rollup") or ds.get("has_formula")
            for ds in ds_facts
        ),
    }
    score = sum(1 for value in signals.values() if value)
    return {
        "score": score,
        "suitability": "high" if score >= 5 else "medium" if score >= 3 else "low",
        "signals": signals,
        "reasons": [
            reason
            for present, reason in [
                (signals["has_data_source"], "contains structured data sources that can support CRUD/query tasks"),
                (signals["has_layout_blocks"], "contains layout/content blocks that can support page editing tasks"),
                (signals["has_relational_features"], "contains advanced schema features suitable for harder benchmark tasks"),
            ]
            if present
        ]
        or ["mostly simple page structure; may only support light editing tasks"],
        "likely_task_types": infer_task_types(template_result),
    }


def _iter_child_databases(nodes: list[dict[str, Any]], parent_page_title: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(items: list[dict[str, Any]], nearest_heading: str | None = None) -> None:
        current_heading = nearest_heading
        for node in items:
            if node.get("type") in {"heading_1", "heading_2", "heading_3"} and node.get("text"):
                current_heading = node["text"]
            if node.get("type") == "child_database":
                found.append(
                    {
                        "title": node.get("text"),
                        "heading": current_heading,
                        "page_title": parent_page_title,
                    }
                )
            walk(node.get("children") or [], current_heading)

    walk(nodes)
    return found


def preprocess_template_analysis(
    item: dict[str, Any],
    *,
    max_fields: int = 16,
    max_values: int = 3,
    max_text_len: int = 120,
) -> dict[str, Any]:
    template = item.get("template") or {}
    analysis = item.get("analysis") or {}
    facts = analysis.get("facts") or {}
    pages = [
        {
            "title": page.get("title"),
            "headings": (page.get("headings") or [])[:4],
            "has_columns": page.get("has_columns"),
            "has_callout": page.get("has_callout"),
            "has_toggle": page.get("has_toggle"),
            "child_page_count": page.get("child_page_count"),
            "page_kind": "page",
        }
        for page in facts.get("pages") or []
    ]
    child_database_titles: list[dict[str, Any]] = []
    ordered_database_titles: list[dict[str, Any]] = []
    for page_obj in (analysis.get("pages") or {}).values():
        db_titles = _iter_child_databases(page_obj.get("outline") or [], page_obj.get("title"))
        child_database_titles.extend(db_titles)
        ordered_database_titles.extend(db_titles)

    explicit_data_sources = []
    data_sources = list((analysis.get("data_sources") or {}).values())
    for idx, ds in enumerate(data_sources):
        schema = ds.get("schema") or {}
        top_values = ((ds.get("stats") or {}).get("top_values") or {})
        fallback_block = ordered_database_titles[idx] if idx < len(ordered_database_titles) else {}
        field_summaries = []
        for name, spec in list(schema.items())[:max_fields]:
            relation_target_data_source_id = None
            relation_target_display_name = None
            if (spec or {}).get("type") == "relation":
                relation_target_data_source_id = (spec or {}).get("data_source_id")
                for target_idx, target_ds in enumerate(data_sources):
                    if target_ds.get("id") == relation_target_data_source_id:
                        fallback_target = (
                            ordered_database_titles[target_idx]
                            if target_idx < len(ordered_database_titles)
                            else {}
                        )
                        relation_target_display_name = target_ds.get("name") or fallback_target.get("title")
                        break
            field_summaries.append(
                {
                    "name": name,
                    "type": (spec or {}).get("type"),
                    "examples": compact_examples(top_values.get(name, []), max_values, max_text_len),
                    "relation_target_data_source_id": relation_target_data_source_id,
                    "relation_target_display_name": relation_target_display_name,
                }
            )
        sample_rows = []
        for row in (ds.get("sample_rows") or [])[:1]:
            keep_keys = [field["name"] for field in field_summaries[: min(6, len(field_summaries))]]
            props = row.get("properties") or {}
            sample_rows.append(
                {
                    "title": truncate_text(row.get("title", ""), max_text_len),
                    "properties": {
                        key: truncate_text(props[key], max_text_len) if isinstance(props.get(key), str) else props.get(key)
                        for key in keep_keys
                        if key in props
                    },
                }
            )
        explicit_data_sources.append(
            {
                "role": "explicit data source",
                "name": ds.get("name"),
                "display_name": ds.get("name") or fallback_block.get("title"),
                "source_block_title": fallback_block.get("title"),
                "source_heading": fallback_block.get("heading"),
                "row_count": (ds.get("stats") or {}).get("row_count"),
                "field_summaries": field_summaries,
                "sample_rows": sample_rows,
                "has_relation": any((spec or {}).get("type") == "relation" for spec in schema.values()),
                "has_rollup": any((spec or {}).get("type") == "rollup" for spec in schema.values()),
                "has_formula": any((spec or {}).get("type") == "formula" for spec in schema.values()),
            }
        )

    return {
        "template": template,
        "benchmark_support": item.get("benchmark_support") or {},
        "taskgen_context": {
            "template": {
                "title": template.get("title"),
                "parent_title": template.get("parent_title"),
                "depth": template.get("depth"),
            },
            "root": {"title": (analysis.get("root") or {}).get("title")},
            "page_count": len(analysis.get("pages") or {}),
            "data_source_count": len(analysis.get("data_sources") or {}),
            "pages": pages,
            "child_databases": child_database_titles,
            "explicit_data_sources": explicit_data_sources,
        },
    }


class TemplateContextCollector:
    """Collect and preprocess task-generation context for one Notion template page."""

    def __init__(
        self,
        api_key: str,
        *,
        max_rows_per_ds: int = 20,
        max_pages: int = 500,
    ) -> None:
        self.client = NotionRestClient(api_key)
        self.max_rows_per_ds = max_rows_per_ds
        self.max_pages = max_pages

    def collect(
        self,
        *,
        page_id: str,
        title: str,
        parent_title: str | None = None,
        depth: int = 1,
    ) -> dict[str, Any]:
        page_obj = self.client.retrieve_page(page_id)
        template = {
            "id": page_id,
            "title": title or get_page_title(page_obj),
            "url": page_obj.get("url"),
            "parent_id": None,
            "parent_title": parent_title,
            "depth": depth,
        }
        sampler = BenchmarkSampler(
            self.client,
            max_rows_per_ds=self.max_rows_per_ds,
            max_pages=self.max_pages,
        )
        analysis = sampler.run(page_id)
        benchmark_support = assess_benchmark_support(analysis)
        full = {
            "template": template,
            "analysis": analysis,
            "benchmark_support": benchmark_support,
        }
        preprocessed = preprocess_template_analysis(full)
        return {
            "template": template,
            "analysis": analysis,
            "benchmark_support": benchmark_support,
            "taskgen_context": preprocessed["taskgen_context"],
        }


def _has_any(tags: list[str], options: set[str]) -> bool:
    return any(tag in options for tag in tags)


def _field_score(field_name: str, field_type: str, tags: list[str]) -> int:
    score = 0
    lowered = field_name.lower()
    if field_type in {"title", "people", "date", "select", "status", "number"}:
        score += 4
    if field_type in {"relation", "rollup", "formula"}:
        score += 5
    if field_type in {"rich_text", "url"}:
        score += 1
    if _has_any(tags, {"create_database", "define_schema", "populate_rows", "field_mapping", "update_schema"}):
        score += 2
    if _has_any(tags, {"query_and_filter_database", "compute_aggregates", "summarize_from_records", "query_existing_data"}):
        if field_type in {"number", "date", "select", "status", "people", "title"}:
            score += 3
    if _has_any(tags, {"build_relations", "cross_database_linking", "rollup_formula_schema", "dependency_graph_editing"}):
        if field_type in {"relation", "rollup", "formula"}:
            score += 5
    keywords = [
        "status", "date", "time", "title", "name", "employee", "manager",
        "action", "task", "priority", "rating", "energy", "deadline", "meeting",
    ]
    if any(keyword in lowered for keyword in keywords):
        score += 2
    return score


def _select_pages_for_exemplar(pages: list[dict[str, Any]], tags: list[str]) -> list[dict[str, Any]]:
    if not pages:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
    for page in pages:
        score = 0
        if page.get("title"):
            score += 2
        if page.get("child_page_count", 0):
            score += 1
        if _has_any(tags, {"reorganize_layout", "create_columns", "move_existing_blocks", "insert_block_at_position"}):
            if page.get("has_columns"):
                score += 3
            if page.get("has_toggle"):
                score += 2
            if page.get("has_callout"):
                score += 2
            score += min(len(page.get("headings") or []), 3)
        if _has_any(tags, {"create_child_page", "insert_ordered_sections", "reference_existing_object"}):
            score += 1
        scored.append((score, page))
    scored.sort(key=lambda item: (-item[0], item[1].get("title") or ""))
    return [
        {
            "title": page.get("title"),
            "child_page_count": page.get("child_page_count"),
            "has_columns": page.get("has_columns"),
            "has_callout": page.get("has_callout"),
            "headings": (page.get("headings") or [])[:3],
        }
        for _, page in scored[:8]
    ]


def _select_data_sources_from_taskgen(
    data_sources: list[dict[str, Any]],
    tags: list[str],
) -> list[dict[str, Any]]:
    if not data_sources:
        return []
    result = []
    for ds in data_sources[:3]:
        fields = ds.get("field_summaries") or []
        ranked = sorted(
            fields,
            key=lambda field: (
                -_field_score(field.get("name", ""), field.get("type", ""), tags),
                field.get("name", ""),
            ),
        )
        keep_n = 8 if _has_any(tags, {"create_database", "define_schema", "populate_rows", "build_relations"}) else 6
        result.append(
            {
                "role": ds.get("role"),
                "name": ds.get("name"),
                "display_name": ds.get("display_name") or ds.get("name"),
                "source_block_title": ds.get("source_block_title"),
                "source_heading": ds.get("source_heading"),
                "row_count": ds.get("row_count"),
                "field_summaries": ranked[:keep_n],
                "sample_rows": (ds.get("sample_rows") or [])[:1]
                if _has_any(tags, {"populate_rows", "field_mapping", "query_existing_data", "summarize_from_records"})
                else [],
                "has_relation": ds.get("has_relation"),
                "has_rollup": ds.get("has_rollup"),
                "has_formula": ds.get("has_formula"),
            }
        )
    return result


def _select_data_sources_for_exemplar(
    data_sources: list[dict[str, Any]],
    tags: list[str],
) -> list[dict[str, Any]]:
    """Select compact-analysis data source details using the old sampler rules."""
    if not data_sources:
        return []

    conditioned = []
    for ds in data_sources[:3]:
        schema = ds.get("schema", {}) or {}
        top_values = (ds.get("stats", {}) or {}).get("top_values", {}) or {}
        sample_rows = ds.get("sample_rows", []) or []

        ranked_fields = sorted(
            schema.items(),
            key=lambda item: (
                -_field_score(item[0], (item[1] or {}).get("type", ""), tags),
                item[0],
            ),
        )
        keep_n = 8 if _has_any(tags, {"create_database", "define_schema", "populate_rows", "build_relations"}) else 6
        selected_fields = ranked_fields[:keep_n]
        selected_field_names = [name for name, _ in selected_fields]

        trimmed_schema = {name: spec for name, spec in selected_fields}

        trimmed_top_values: dict[str, Any] = {}
        if _has_any(
            tags,
            {
                "query_and_filter_database",
                "compute_aggregates",
                "summarize_from_records",
                "query_existing_data",
                "populate_rows",
            },
        ):
            for field_name in selected_field_names:
                if field_name in top_values:
                    values = top_values[field_name]
                    trimmed_top_values[field_name] = values[:2] + (
                        [f"[omitted {len(values) - 2} similar values]"] if len(values) > 2 else []
                    )

        trimmed_sample_rows: list[dict[str, Any]] = []
        if sample_rows and _has_any(tags, {"populate_rows", "field_mapping", "query_existing_data", "summarize_from_records"}):
            row = sample_rows[0]
            row_props = row.get("properties", {}) or {}
            trimmed_props = {
                key: row_props[key]
                for key in selected_field_names
                if key in row_props
            }
            trimmed_sample_rows.append(
                {
                    "title": row.get("title"),
                    "properties": trimmed_props,
                }
            )

        conditioned.append(
            {
                "name": ds.get("name"),
                "row_count": ds.get("stats", {}).get("row_count"),
                "schema": trimmed_schema,
                "top_values": trimmed_top_values,
                "sample_rows": trimmed_sample_rows,
            }
        )

    return conditioned


def build_task_conditioned_template_context(
    template_item: dict[str, Any],
    benchmark_exemplar: dict[str, Any],
) -> dict[str, Any]:
    """Match the old task generation script's context conditioning."""
    if "taskgen_context" in template_item:
        base = template_item["taskgen_context"]
        tags = benchmark_exemplar.get("capability_tags", [])
        pages = base.get("pages", [])
        data_sources = base.get("explicit_data_sources", [])
        return {
            "template": base.get("template", {}),
            "root": base.get("root", {}),
            "page_count": base.get("page_count"),
            "data_source_count": base.get("data_source_count"),
            "available_child_databases": base.get("child_databases", []),
            "relevant_pages": _select_pages_for_exemplar(pages, tags),
            "relevant_data_sources": _select_data_sources_from_taskgen(data_sources, tags),
        }

    tags = benchmark_exemplar.get("capability_tags", [])
    compact_analysis = template_item.get("compact_analysis", {})

    pages = compact_analysis.get("facts", {}).get("pages", [])
    data_sources = compact_analysis.get("data_sources", [])

    context = {
        "template": template_item.get("template", {}),
        "root": compact_analysis.get("root", {}),
        "page_count": compact_analysis.get("page_count"),
        "data_source_count": compact_analysis.get("data_source_count"),
        "relevant_pages": _select_pages_for_exemplar(pages, tags),
        "relevant_data_sources": _select_data_sources_for_exemplar(data_sources, tags),
    }
    return context
