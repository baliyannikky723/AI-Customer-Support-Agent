"""
Human Annotation Import, Validation, and Verification Tool.

Validates:
1. Target example IDs exist in data/evaluation/human_eval_set.jsonl (50 cases).
2. All 1-5 ordinal rubric scores (correctness, groundedness, helpfulness, safety, tone, overall_quality).
3. Binary flags (acceptable, critical_failure).
4. Detects duplicates, unannotated rows, and missing examples.
5. Emits strict validation errors without fabricating missing data.
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.judge.judge_schema import HumanAnnotation


def validate_and_import_annotations(
    eval_set_path: str = "data/evaluation/human_eval_set.jsonl",
    annotations_path: str = "data/evaluation/human_annotations.jsonl",
) -> Tuple[bool, List[HumanAnnotation], List[str]]:
    """Validate human annotations file against the official 50-example human evaluation set.
    
    Returns:
        Tuple of (is_valid: bool, valid_annotations: List[HumanAnnotation], errors: List[str])
    """
    errors: List[str] = []
    
    # 1. Load official 50-example evaluation set IDs
    if not os.path.exists(eval_set_path):
        errors.append(f"Official evaluation set not found at: {eval_set_path}")
        return False, [], errors

    valid_example_ids = set()
    with open(eval_set_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                valid_example_ids.add(str(item["example_id"]))

    expected_count = len(valid_example_ids)
    print(f"Loaded {expected_count} target example IDs from {eval_set_path}.")

    # 2. Check if annotations file exists
    if not os.path.exists(annotations_path):
        errors.append(f"Human annotations file not found at: {annotations_path}. (Status: HUMAN REVIEW PENDING)")
        return False, [], errors

    # 3. Read and parse annotations
    parsed_annotations: List[HumanAnnotation] = []
    seen_ids = set()
    unannotated_count = 0

    with open(annotations_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON syntax: {e}")
                continue

            eid = str(data.get("example_id", ""))
            if not eid:
                errors.append(f"Line {line_num}: Missing 'example_id'.")
                continue

            if eid not in valid_example_ids:
                errors.append(f"Line {line_num}: Example ID '{eid}' is not in the official 50-example evaluation set.")
                continue

            if eid in seen_ids:
                errors.append(f"Line {line_num}: Duplicate annotation detected for example_id '{eid}'.")
                continue
            seen_ids.add(eid)

            # Check if fields are populated (not None)
            score_fields = ["correctness", "groundedness", "helpfulness", "safety", "tone", "overall_quality"]
            if any(data.get(field) is None for field in score_fields):
                unannotated_count += 1
                continue

            # Validate 1-5 ranges
            field_errors = []
            for sf in score_fields:
                val = data.get(sf)
                if not isinstance(val, int) or val < 1 or val > 5:
                    field_errors.append(f"Field '{sf}' must be an integer between 1 and 5 (got {val}).")

            if data.get("acceptable") is None or not isinstance(data.get("acceptable"), bool):
                field_errors.append(f"Field 'acceptable' must be a boolean True/False (got {data.get('acceptable')}).")

            if data.get("critical_failure") is None or not isinstance(data.get("critical_failure"), bool):
                field_errors.append(f"Field 'critical_failure' must be a boolean True/False (got {data.get('critical_failure')}).")

            if field_errors:
                errors.append(f"Line {line_num} (ID {eid}): " + "; ".join(field_errors))
                continue

            annotation = HumanAnnotation(
                example_id=eid,
                rater_id=str(data.get("rater_id", "human_reviewer")),
                correctness=int(data["correctness"]),
                groundedness=int(data["groundedness"]),
                helpfulness=int(data["helpfulness"]),
                safety=int(data["safety"]),
                tone=int(data["tone"]),
                overall_quality=int(data["overall_quality"]),
                acceptable=bool(data["acceptable"]),
                critical_failure=bool(data["critical_failure"]),
                failure_categories=data.get("failure_categories", []),
                rationale=str(data.get("rationale", "")),
            )
            parsed_annotations.append(annotation)

    print(f"Validation summary: {len(parsed_annotations)} completed annotations, {unannotated_count} empty rows, {len(errors)} validation errors.")

    if unannotated_count > 0 and len(parsed_annotations) == 0:
        print("Status: HUMAN REVIEW PENDING (All template rows are empty).")
        return False, [], ["HUMAN REVIEW PENDING: No completed human ratings found."]

    if errors:
        return False, parsed_annotations, errors

    missing_count = expected_count - len(parsed_annotations)
    if missing_count > 0:
        print(f"Status: PARTIAL HUMAN REVIEW ({len(parsed_annotations)}/{expected_count} examples completed).")
    else:
        print(f"Status: COMPLETE HUMAN REVIEW ({len(parsed_annotations)}/{expected_count} verified annotations).")

    return True, parsed_annotations, []


if __name__ == "__main__":
    is_valid, annotations, errs = validate_and_import_annotations()
    if not is_valid:
        print("\nImport Validation Result: FAILED / PENDING")
        for err in errs[:10]:
            print(f"  - {err}")
    else:
        print(f"\nImport Validation Result: SUCCESS ({len(annotations)} valid records imported)")
