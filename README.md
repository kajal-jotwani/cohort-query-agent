# Cohort Query Agent Prototype

This repository contains a small local prototype for two pieces of the cohort query flow:

1. Schema linking from natural language to PCDC fields
2. Filter validation before sending a query to Guppy

Everything runs locally and does not require API keys.

## What is included

### schema_linker.py

`SchemaLinker` builds a TF-IDF index over schema field descriptions and uses cosine similarity to find likely field matches for a query.

- Clear matches are returned directly
- Uncertain matches are marked as ambiguous so they can be handled in one batched LLM call later

```python
from schema_linker import SchemaLinker

linker = SchemaLinker()
result = linker.link("INRG patients with metastatic bone tumors")
print(result.summary())
```

### filter_validator.py

`validate_and_report()` checks filter structure and values against the schema before execution.

It validates:

- field names
- enum values
- operator usage (`IN` vs `GTE`/`LTE`)
- nested table paths
- field/table alignment inside nested blocks

```python
from filter_validator import validate_and_report

is_valid, report = validate_and_report({
  "AND": [{"IN": {"tumor_stage": ["Metastatic"]}}]
})
print(report)
```

## Sample benchmark output

From `benchmark.py`:

```text
Schema linker avg recall   : 89.6%  (15/20 perfect field matches)
Schema linker avg latency  : 1.03ms per query
Validator accuracy         : 100%   (8/8 test cases)
LLM calls eliminated       : 111 / 149 total old calls across 20 queries
Speedup factor             : ~3.9x fewer LLM calls
```

Some lower-recall queries are age-range phrased prompts (for example "under 5 years old"), where similarly named age fields compete.

## Run locally

```bash
pip install numpy scikit-learn
python benchmark.py
```

Quick linker check:

```bash
python -c "
from schema_linker import SchemaLinker
linker = SchemaLinker()
print(linker.link('INRG patients with metastatic bone tumors').summary())
"
```

Quick validator check:

```bash
python -c "
from filter_validator import validate_and_report
ok, report = validate_and_report({'AND': [{'IN': {'tumor_stage': ['Metastatic']}}]})
print(report)
"
```

## Scope notes

This is intentionally a compact prototype. It does not include:

- full LLM generation + routing flow
- full production schema coverage
- FAISS indexing (uses sklearn cosine similarity for portability)
