"""
filter_validator.py
Pre-execution schema validator for LLM-generated GraphQL filters.

currently the filter goes straight to Guppy API with no
validation. Hallucinated field names only surface as cryptic API errors.

This module runs BEFORE the filter reaches Guppy and returns structured
errors that the UI can display and optionally feed back to the LLM for
self-correction.

Operators validated:
    IN     - field must exist, must be enum type, values must be in allowed set
    GTE    - field must exist, must be numeric type
    LTE    - field must exist, must be numeric type
    nested - path must exist as a known table, all child fields must belong to that table
    AND    - recursive validation of all clauses
"""

from schema_linker import PCDC_FIELDS, GITOPS_MAP
from difflib import get_close_matches

# Build fast lookup structures
_FIELD_BY_NAME  = {f["field"]: f for f in PCDC_FIELDS}
_ALL_FIELD_NAMES = list(_FIELD_BY_NAME.keys())

# Valid table paths from processed_gitops.json (PR #5)
# Only paths that actually appear as non-empty lists
_VALID_TABLES = {
    "tumor_assessments", "survival_characteristics", "subject_responses",
    "molecular_analysis", "histologies", "stagings", "labs",
    "biopsy_surgical_procedures", "radiation_therapies", "studies",
    "medical_histories", "external_references", "imagings",
    "stem_cell_transplants", "secondary_malignant_neoplasm",
    "disease_characteristics",
}


class ValidationError:
    def __init__(self, path: str, message: str, suggestion: str = ""):
        self.path       = path
        self.message    = message
        self.suggestion = suggestion

    def __str__(self):
        s = f"[{self.path}] {self.message}"
        if self.suggestion:
            s += f"\n         Suggestion: {self.suggestion}"
        return s


def _suggest_field(name: str) -> str:
    """Fuzzy-match a bad field name against real ones."""
    matches = get_close_matches(name, _ALL_FIELD_NAMES, n=1, cutoff=0.6)
    return f'did you mean "{matches[0]}"?' if matches else "check pcdc-schema-prod-20250114.json"


def validate_filter(filter_obj: dict, path: str = "root") -> list[ValidationError]:
    """
    Recursively validate a PCDC GraphQL filter object.
    Returns a list of ValidationError. Empty list = filter is valid.
    """
    errors = []

    if not isinstance(filter_obj, dict):
        errors.append(ValidationError(path, f"expected object, got {type(filter_obj).__name__}"))
        return errors

    for operator, value in filter_obj.items():

        op_path = f"{path}.{operator}"

        # AND 
        if operator == "AND":
            if not isinstance(value, list):
                errors.append(ValidationError(op_path, "AND must be an array"))
                continue
            for i, clause in enumerate(value):
                errors.extend(validate_filter(clause, f"{op_path}[{i}]"))

        # IN 
        elif operator == "IN":
            if not isinstance(value, dict):
                errors.append(ValidationError(op_path, "IN must map field_name -> [values]"))
                continue
            for field_name, field_values in value.items():
                fpath = f"{op_path}.{field_name}"
                schema = _FIELD_BY_NAME.get(field_name)

                if schema is None:
                    errors.append(ValidationError(
                        fpath,
                        f'unknown field "{field_name}"',
                        _suggest_field(field_name)
                    ))
                    continue

                if schema["type"] == "number":
                    errors.append(ValidationError(
                        fpath,
                        f'"{field_name}" is a numeric field - use GTE/LTE, not IN',
                        f'replace with {{"GTE": {{"{field_name}": <value>}}}}'
                    ))
                    continue

                if not isinstance(field_values, list):
                    errors.append(ValidationError(fpath, "IN values must be an array"))
                    continue

                valid_enums_lower = {e.lower() for e in schema["enums"]}
                for v in field_values:
                    if v.lower() not in valid_enums_lower:
                        close = get_close_matches(v, schema["enums"], n=1, cutoff=0.6)
                        suggestion = f'did you mean "{close[0]}"?' if close else f'valid values: {schema["enums"]}'
                        errors.append(ValidationError(
                            fpath,
                            f'invalid enum value "{v}" for field "{field_name}"',
                            suggestion
                        ))

        # GTE / LTE 
        elif operator in ("GTE", "LTE"):
            if not isinstance(value, dict):
                errors.append(ValidationError(op_path, f"{operator} must map field_name -> numeric_value"))
                continue
            for field_name, field_value in value.items():
                fpath = f"{op_path}.{field_name}"
                schema = _FIELD_BY_NAME.get(field_name)

                if schema is None:
                    errors.append(ValidationError(fpath, f'unknown field "{field_name}"', _suggest_field(field_name)))
                    continue

                if schema["type"] != "number":
                    errors.append(ValidationError(
                        fpath,
                        f'"{field_name}" is an enum field - use IN, not {operator}',
                        f'replace with {{"IN": {{"{field_name}": [<values>]}}}}'
                    ))

                if not isinstance(field_value, (int, float)):
                    errors.append(ValidationError(fpath, f"{operator} value must be a number, got {type(field_value).__name__}"))

        #  nested 
        elif operator == "nested":
            if not isinstance(value, dict):
                errors.append(ValidationError(op_path, "nested must be an object with 'path' and 'AND'"))
                continue

            path_val = value.get("path")
            if path_val is None:
                errors.append(ValidationError(op_path, "nested block missing required 'path' key"))
            elif path_val not in _VALID_TABLES:
                close = get_close_matches(path_val, list(_VALID_TABLES), n=1, cutoff=0.6)
                suggestion = f'did you mean "{close[0]}"?' if close else f"valid paths: {sorted(_VALID_TABLES)}"
                errors.append(ValidationError(op_path, f'unknown nested path "{path_val}"', suggestion))
            else:
                # All fields inside this nested block must belong to path_val table
                if "AND" in value:
                    for i, clause in enumerate(value["AND"]):
                        child_errors = validate_filter(clause, f"{op_path}.AND[{i}]")
                        # Check field belongs to the stated nested path
                        for op2, v2 in clause.items():
                            if op2 == "IN" and isinstance(v2, dict):
                                for fn in v2.keys():
                                    allowed_paths = GITOPS_MAP.get(fn)
                                    field_schema = _FIELD_BY_NAME.get(fn)
                                    if allowed_paths is None or field_schema is None:
                                        continue  # unknown field - already caught above
                                    # Empty list means subject-level (not nested)
                                    if len(allowed_paths) == 0:
                                        child_errors.append(ValidationError(
                                            f"{op_path}.AND[{i}].IN.{fn}",
                                            f'field "{fn}" is subject-level - move it outside the nested block',
                                            f'remove from nested.{path_val} and add to the top-level AND'
                                        ))
                                    elif path_val not in allowed_paths:
                                        child_errors.append(ValidationError(
                                            f"{op_path}.AND[{i}].IN.{fn}",
                                            f'field "{fn}" belongs to {allowed_paths}, not "{path_val}"',
                                            f'change path to "{allowed_paths[0]}" or move field out'
                                        ))
                        errors.extend(child_errors)

        #  unknown operator
        elif operator not in ("path",):   # "path" is a known nested key, not an operator
            errors.append(ValidationError(op_path, f'unknown operator "{operator}" - valid: AND, IN, GTE, LTE, nested'))

    return errors


def validate_and_report(filter_obj: dict) -> tuple[bool, str]:
    """
    Convenience wrapper. Returns (is_valid, report_string).
    """
    errors = validate_filter(filter_obj)
    if not errors:
        return True, "OK Filter is valid - all fields, types, and enum values confirmed against PCDC schema"

    lines = [f"X {len(errors)} validation error(s) found:\n"]
    for e in errors:
        lines.append(f"  * {e}")
    return False, "\n".join(lines)
