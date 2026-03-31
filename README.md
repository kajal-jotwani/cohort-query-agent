# PCDC Schema Linker - Working Prototype
### GSoC 2025 Proposal - Enhancing the Cohort Discovery Chatbot

This is a **working prototype** of two core components proposed in my GSoC 2026
application. Both run locally with no API keys



## What this demonstrates

### 1. `schema_linker.py` - Replaces the N+1 LLM loop (fixes Issue #9)

The current codebase in `nested_graphql_helper.py` makes one GPT-4o call per
ambiguous keyword during schema disambiguation:

```python
# Current code - the problem
for keyword in keywords:
    result = await query_processed_pcdc_result(keyword, candidates)
    # ^ one sequential LLM call per term = 30+ second total latency
```

`SchemaLinker` replaces this with a TF-IDF cosine similarity index built
over the PCDC schema at startup. Clear matches resolve in under 2ms with
zero LLM calls. Only genuinely ambiguous pairs go to a single batched call.

```python
# Proposed replacement
linker = SchemaLinker()
result = linker.link("INRG patients with metastatic bone tumors")
# result.direct_hits  -> resolved by vector search, no LLM
# result.ambiguous    -> goes to ONE batched LLM call
```

### 2. `filter_validator.py` - Pre-execution schema validation (fixes Issue #14)

The current system sends LLM-generated filters straight to the Guppy API.
Hallucinated field names produce cryptic API errors.

`validate_and_report()` checks every field name, enum value, operator type,
and nested path against the PCDC schema before the query runs:

```python
is_valid, report = validate_and_report(generated_filter)
# Catches: unknown fields, wrong enum values, numeric fields with IN,
#          wrong nested paths, cross-table field contamination
```

---

## Benchmark results (from running benchmark.py)

```
Schema linker avg recall   : 89.6%  (15/20 perfect field matches)
Schema linker avg latency  : 1.03ms per query
Validator accuracy         : 100%   (8/8 test cases)
LLM calls eliminated       : 111 / 149 total old calls across 20 queries
Speedup factor             : ~3.9x fewer LLM calls
```

The 4 partial-recall cases (50-75% recall) are queries involving numeric age
ranges like "under 5 years old" - the TF-IDF model conflates
`age_at_enrollment` (subject-level) with `age_at_tumor_assessment`
(tumor_assessments table). This is the exact ambiguity that motivates the
batched LLM fallback in the proposed design.

---

## How to run

```bash
# Requirements: Python 3.11+, numpy, scikit-learn
pip install numpy scikit-learn

# Run the full benchmark (20 linker queries + 8 validator cases)
python benchmark.py

# Try the linker interactively
python -c "
from schema_linker import SchemaLinker
linker = SchemaLinker()
result = linker.link('your query here')
print(result.summary())
"

# Try the validator interactively
python -c "
from filter_validator import validate_and_report
ok, report = validate_and_report({
    'AND': [
        {'IN': {'tumor_stage': ['Metastatic']}}  # hallucinated field name
    ]
})
print(report)
"
```

## What's not in this prototype

- The LLM generation step (needs OpenAI/Anthropic API key)
- The LangChain agent / intent router (Deliverable C)
- The full PCDC schema (this uses a representative 14-field subset;
  the real schema has ~200+ fields, which would improve recall further)
- FAISS index (replaced here with sklearn cosine_similarity for portability;
  the real implementation uses FAISS for sub-millisecond search at scale)
