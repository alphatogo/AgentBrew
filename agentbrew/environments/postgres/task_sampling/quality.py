"""Compatibility wrapper around the migrated PostgreSQL generation validator."""

from __future__ import annotations

from typing import Any

from .generation import (
    parse_existing_refs,
    parse_existing_tables,
    parse_table_stats,
    try_parse_json_object,
    validate_generation_payload,
)


def process_generation(
    response: str,
    metadata: dict[str, Any],
    task_type_id: str = "bulk_data_migration",
) -> dict[str, Any]:
    """Parse and validate with the exact migrated task-generation rules."""
    enriched = dict(metadata)
    enriched.setdefault("_table_names", sorted(parse_existing_tables(enriched)))
    enriched.setdefault("_ref_entities", sorted(parse_existing_refs(enriched)))
    enriched.setdefault("_table_stats", parse_table_stats(enriched))
    try:
        payload = validate_generation_payload(
            try_parse_json_object(response), task_type_id, enriched
        )
        return {
            "accepted": True,
            "question": payload["question"],
            "generation": payload,
            "raw_response": response,
            "judgment": {"accepted": True, "hard_reject_reasons": []},
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {
            "accepted": False,
            "question": None,
            "raw_response": response,
            "judgment": {
                "accepted": False,
                "hard_reject_reasons": [str(exc)],
            },
        }
