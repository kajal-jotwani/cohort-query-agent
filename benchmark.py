"""
benchmark.py

Simple benchmark runner for the schema linker and filter validator.

It reports:
    - field retrieval recall
    - per-query latency
    - rough LLM-call savings compared to the older flow
    - validator pass/fail correctness
"""

import time
import json
from schema_linker import SchemaLinker
from filter_validator import validate_and_report


# Ground truth dataset
# Format: (natural_language_query, expected_pcdc_fields, query_type)
# expected_pcdc_fields = fields expected for a reasonable match
GROUND_TRUTH = [
    # All field names sourced from processed_gitops.json in PR #5
    (
        "Show me INRG patients with metastatic tumors",
        ["consortium", "tumor_classification"],
        "enum_nested"
    ),
    (
        "Female patients from INSTRuCT with bone tumor site",
        ["sex", "consortium", "tumor_site"],
        "enum_nested"
    ),
    (
        "INRG patients aged under 3650 days at tumor assessment",
        ["consortium", "age_at_tumor_assessment"],
        "enum_numeric"
    ),
    (
        "Patients with relapsed disease from NODAL consortium",
        ["disease_phase", "consortium"],
        "enum_nested"  # disease_phase is the TODO case
    ),
    (
        "Alive patients from HIBISCUS consortium",
        ["lkss", "consortium"],
        "enum_enum"
    ),
    (
        "INRG patients with absent tumor state on skin",
        ["consortium", "tumor_state", "tumor_site"],
        "enum_nested"
    ),
    (
        "Patients with metastatic localized tumor classification",
        ["tumor_classification"],
        "enum_nested"
    ),
    (
        "Dead patients from HIBISCUS consortium",
        ["lkss", "consortium"],
        "enum_enum"
    ),
    (
        "INTERACT patients with bone tumor site",
        ["consortium", "tumor_site"],
        "enum_nested"
    ),
    (
        "Patients with complete treatment response",
        ["response"],
        "enum_nested"
    ),
    (
        "Asian female patients from INRG",
        ["race", "sex", "consortium"],
        "multi_enum"
    ),
    (
        "Patients with brain tumor site from MaGIC consortium",
        ["tumor_site", "consortium"],
        "enum_nested"
    ),
    (
        "INRG patients with age at tumor assessment under 3650 days",
        ["age_at_tumor_assessment", "consortium"],
        "enum_numeric"
    ),
    (
        "White male patients from NODAL with regional tumor",
        ["race", "sex", "consortium", "tumor_classification"],
        "multi_enum"
    ),
    (
        "Alive patients with metastatic bone tumor",
        ["lkss", "tumor_classification", "tumor_site"],
        "multi_nested"
    ),
    (
        "INSTRuCT females with present tumor state",
        ["consortium", "sex", "tumor_state"],
        "enum_nested"
    ),
    (
        "Patients with progressive disease and partial response",
        ["disease_phase", "response"],
        "enum_nested"  # disease_phase TODO case
    ),
    (
        "Patients with liver tumor site from ALL consortium",
        ["tumor_site", "consortium"],
        "enum_nested"
    ),
    (
        "Hispanic patients from MaGIC with invasive tumor",
        ["ethnicity", "consortium", "invasiveness"],
        "multi_nested"
    ),
    (
        "INRG patients with metastatic bone tumor absent state",
        ["consortium", "tumor_classification", "tumor_site", "tumor_state"],
        "complex_multi"
    ),
]

# Validation test cases
# (description, filter_object, should_be_valid)
VALIDATION_CASES = [
    # Field names sourced from processed_gitops.json in PR #5
    (
        "Valid: simple subject-level filter",
        {"AND": [{"IN": {"consortium": ["INRG"]}}, {"IN": {"sex": ["Male"]}}]},
        True
    ),
    (
        "Valid: correct nested filter (matches PR #5 README example)",
        {"AND": [
            {"IN": {"consortium": ["INRG"]}},
            {"nested": {
                "path": "tumor_assessments",
                "AND": [
                    {"IN": {"tumor_classification": ["Metastatic"]}},
                    {"IN": {"tumor_state": ["Absent"]}},
                    {"IN": {"tumor_site": ["Skin"]}}
                ]
            }}
        ]},
        True
    ),
    (
        "Valid: numeric GTE/LTE with real field age_at_tumor_assessment",
        {"nested": {
            "path": "tumor_assessments",
            "AND": [{"GTE": {"age_at_tumor_assessment": 0}}, {"LTE": {"age_at_tumor_assessment": 3650}}]
        }},
        True
    ),
    (
        "Valid: survival_characteristics nested with real field lkss",
        {"nested": {
            "path": "survival_characteristics",
            "AND": [{"IN": {"lkss": ["Alive"]}}]
        }},
        True
    ),
    (
        "INVALID: hallucinated field 'vital_status' - real field is 'lkss'",
        {"AND": [{"IN": {"vital_status": ["Alive"]}}]},
        False
    ),
    (
        "INVALID: hallucinated 'age_at_enrollment' - real field is 'age_at_censor_status'",
        {"AND": [{"IN": {"age_at_enrollment": [5]}}]},
        False
    ),
    (
        "INVALID: numeric field age_at_tumor_assessment used with IN operator",
        {"nested": {
            "path": "tumor_assessments",
            "AND": [{"IN": {"age_at_tumor_assessment": [365, 730]}}]
        }},
        False
    ),
    (
        "INVALID: wrong nested path 'tumors' - real path is 'tumor_assessments'",
        {"nested": {"path": "tumors", "AND": [{"IN": {"tumor_site": ["Bone"]}}]}},
        False
    ),
]


def run_linker_benchmark(linker: SchemaLinker) -> dict:
    """Run all ground-truth queries and measure field retrieval accuracy and latency."""
    results = []

    for query, expected_fields, qtype in GROUND_TRUTH:
        t_start = time.perf_counter()
        result = linker.link(query)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        retrieved = {h["field"] for h in result.direct_hits + result.ambiguous}
        expected  = set(expected_fields)

        # Recall: what fraction of expected fields were retrieved
        recall = len(retrieved & expected) / len(expected) if expected else 1.0

        # Old approach: 1 LLM call per term + 1 generation = N+1
        tokens   = [w for w in query.lower().split() if len(w) > 3]
        old_calls = len(tokens) + 1

        # New approach: 0 or 1 disambiguation call + 1 generation
        new_calls = (1 if result.needs_llm_batch() else 0) + 1

        results.append({
            "query":          query,
            "type":           qtype,
            "recall":         recall,
            "retrieved":      sorted(retrieved),
            "expected":       sorted(expected),
            "vector_ms":      round(elapsed_ms, 2),
            "direct_hits":    len(result.direct_hits),
            "ambiguous":      len(result.ambiguous),
            "old_llm_calls":  old_calls,
            "new_llm_calls":  new_calls,
            "calls_saved":    old_calls - new_calls,
        })

    return results


def run_validator_benchmark() -> dict:
    """Run all validation test cases and check correctness."""
    results = []
    for desc, filter_obj, should_be_valid in VALIDATION_CASES:
        t_start = time.perf_counter()
        is_valid, report = validate_and_report(filter_obj)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        correct = (is_valid == should_be_valid)
        results.append({
            "desc":        desc,
            "expected":    "VALID" if should_be_valid else "INVALID",
            "got":         "VALID" if is_valid else "INVALID",
            "correct":     correct,
            "report":      report,
            "elapsed_ms":  round(elapsed_ms, 3),
        })

    return results


def print_report(linker_results: list, validator_results: list):
    W = 72

    def box(title):
        print(f"  {title}")


    box("Schema linker benchmark (20 queries)")
    print(f"  {'Query (truncated)':<42} {'Type':<14} {'Recall':<8} {'ms':<7} {'Saved'}")
    print(f"  {'-'*42} {'-'*14} {'-'*8} {'-'*7} {'-'*5}")

    recalls, latencies, calls_saved = [], [], []
    for r in linker_results:
        q_short = r["query"][:40] + ".." if len(r["query"]) > 40 else r["query"]
        recall_str = f"{r['recall']:.0%}"
        mark = "Good" if r["recall"] >= 0.8 else ("Fair" if r["recall"] >= 0.5 else "Low")
        print(f"  {q_short:<42} {r['type']:<14} {mark} {recall_str:<6} {r['vector_ms']:<7.1f} -{r['calls_saved']}")
        recalls.append(r["recall"])
        latencies.append(r["vector_ms"])
        calls_saved.append(r["calls_saved"])

    avg_recall   = sum(recalls) / len(recalls)
    avg_latency  = sum(latencies) / len(latencies)
    total_saved  = sum(calls_saved)
    perfect      = sum(1 for r in recalls if r == 1.0)

    print(f"\n  Average recall        : {avg_recall:.1%}  ({perfect}/{len(recalls)} perfect matches)")
    print(f"  Average vector latency: {avg_latency:.2f}ms per query")
    print(f"  Total LLM calls saved : {total_saved} calls across {len(linker_results)} queries")
    print(f"  Estimated time saved (assuming ~1400ms/call): {total_saved * 1.4:.1f}s")

    # Breakdown by type
    by_type = {}
    for r in linker_results:
        by_type.setdefault(r["type"], []).append(r["recall"])
    print(f"\n  Recall by query type:")
    for qtype, recs in sorted(by_type.items()):
        print(f"    {qtype:<20} {sum(recs)/len(recs):.0%}  ({len(recs)} queries)")

    box("Filter validator benchmark (8 cases)")
    print(f"  {'Test case':<52} {'Expected':<10} {'Got':<10} {'Pass?'}")

    correct_count = 0
    for r in validator_results:
        desc_short = r["desc"][:50] + ".." if len(r["desc"]) > 50 else r["desc"]
        mark = "Pass" if r["correct"] else "Fail"
        print(f"  {desc_short:<52} {r['expected']:<10} {r['got']:<10} {mark}")
        if r["correct"]:
            correct_count += 1
        elif not r["correct"]:
            # Print what the validator actually said for failures
            short_report = r["report"][:120].replace("\n", " ")
            print(f"    -> {short_report}")

    print(f"\n  Validator accuracy: {correct_count}/{len(validator_results)} ({correct_count/len(validator_results):.0%})")
    print(f"  False positives (valid rejected): {sum(1 for r in validator_results if not r['correct'] and r['expected']=='VALID')}")
    print(f"  False negatives (invalid passed): {sum(1 for r in validator_results if not r['correct'] and r['expected']=='INVALID')}")

    box("Summary")
    print(f"  Schema linker avg recall  : {avg_recall:.1%}")
    print(f"  Schema linker avg latency : {avg_latency:.2f}ms  (vs ~1200-2000ms per LLM call)")
    print(f"  Validator accuracy        : {correct_count/len(validator_results):.0%}")
    print(f"  LLM calls eliminated      : {total_saved} / {sum(r['old_llm_calls'] for r in linker_results)} total old calls")
    print(f"  Speedup factor            : ~{sum(r['old_llm_calls'] for r in linker_results) / sum(r['new_llm_calls'] for r in linker_results):.1f}x fewer LLM calls")
    print(f"{'-'*W}\n")


if __name__ == "__main__":
    print("\nBuilding TF-IDF index...")
    t0 = time.perf_counter()
    linker = SchemaLinker()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"Index built in {build_ms:.1f}ms ({len(linker.fields)} fields, one-time startup cost)\n")

    linker_results    = run_linker_benchmark(linker)
    validator_results = run_validator_benchmark()
    print_report(linker_results, validator_results)

    # Also show a detailed trace for one interesting query
    print("\nDetailed trace for one multi-condition query:")
    demo = linker.link("INRG patients with metastatic bone disease at initial diagnosis")
    print(demo.summary())
