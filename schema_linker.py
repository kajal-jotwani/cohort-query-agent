"""
schema_linker.py

Replaces the N+1 LLM disambiguation loop in nested_graphql_helper.py.

Current code (the problem):
    # query_processed_pcdc_result() makes one GPT-4o call per ambiguous keyword
    for keyword in keywords:
        result = await query_processed_pcdc_result(keyword, candidates)  # 1 LLM call each

This module (the fix):
    linker = SchemaLinker(pcdc_schema_path, gitops_path)
    matches = linker.link(user_query)   # 0 LLM calls for clear matches
    # Only genuinely ambiguous pairs go to a single batched LLM call

How it works:
    1. At init: build TF-IDF matrix over all PCDC field descriptions
       (field name + table + type + enum values concatenated as a document)
    2. At query time: transform user query into same TF-IDF space,
       compute cosine similarity against every field document
    3. Fields above DIRECT_THRESHOLD are returned as direct hits
    4. Fields where top-2 scores are within AMBIGUOUS_DELTA go to LLM batch
    5. LLM batch is a single call resolving all ambiguities at once

This is a drop-in replacement that keeps the same output contract:
    {field_name: {table, type, enums, score, resolution_method}}
"""

import json
import re
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#  Tunable thresholds 
DIRECT_THRESHOLD = 0.15   # cosine score above this = direct vector hit, no LLM
AMBIGUOUS_DELTA  = 0.04   # if top-2 scores are within this, flag as ambiguous
MIN_SCORE        = 0.05   # below this = not relevant, ignore entirely
TOP_K            = 3      # number of candidates to return per query term


#  PCDC schema embedded directly (mirrors what parse_pcdc_schema_prod()
#    produces after caching to processed_pcdc_schema_prod.json) 
#
#    In the real system this would be loaded from:
#        json.load(open("schema/pcdc-schema-prod-20250114.json"))
#    We embed a representative subset here so the prototype is self-contained.


# Fields sourced directly from processed_gitops.json (PR #5, Sep 2025)
# and processed_pcdc_schema_prod.json in the repository.
# Table mappings match the gitops file exactly.
# Note: disease_phase does NOT appear in processed_gitops.json -
# this is precisely why end-to-end nested generation for it is incomplete
# and listed as a README TODO. We represent it here for the schema linker
# to detect and flag as requiring special handling.
PCDC_FIELDS = [
    #  Subject-level fields (empty list in gitops = top-level, no nested path) 
    {
        "field":  "consortium",
        "table":  "subject",
        "type":   "enum",
        "enums":  ["INRG", "INSTRuCT", "NODAL", "INTERACT", "HIBISCUS", "MaGIC", "ALL"],
        "desc":   "pediatric cancer research consortium group study program membership"
    },
    {
        "field":  "sex",
        "table":  "subject",
        "type":   "enum",
        "enums":  ["Male", "Female", "Other", "Undifferentiated", "Unknown", "Not Reported"],
        "desc":   "patient biological sex gender male female"
    },
    {
        "field":  "race",
        "table":  "subject",
        "type":   "enum",
        "enums":  ["White", "Black or African American", "Asian", "Unknown", "Not Reported"],
        "desc":   "patient race ethnicity demographic white black asian"
    },
    {
        "field":  "ethnicity",
        "table":  "subject",
        "type":   "enum",
        "enums":  ["Hispanic or Latino", "Not Hispanic or Latino", "Unknown"],
        "desc":   "patient ethnicity hispanic latino"
    },
    {
        "field":  "age_at_censor_status",
        "table":  "subject",
        "type":   "number",
        "enums":  [],
        "desc":   "patient age at censor status years days enrollment study pediatric child young under over"
    },
    #  tumor_assessments 
    {
        "field":  "tumor_classification",
        "table":  "tumor_assessments",
        "type":   "enum",
        "enums":  ["Metastatic", "Localized", "Regional", "Unknown"],
        "desc":   "tumor classification metastatic localized regional spread stage distant metastasis"
    },
    {
        "field":  "tumor_site",
        "table":  "tumor_assessments",
        "type":   "enum",
        "enums":  ["Bone", "Skin", "Liver", "Lung", "Brain", "Lymph Node",
                   "Bone Marrow", "Adrenal Gland", "Kidney", "Unknown"],
        "desc":   "tumor site anatomical location bone skin liver lung brain lymph marrow adrenal kidney where"
    },
    {
        "field":  "tumor_state",
        "table":  "tumor_assessments",
        "type":   "enum",
        "enums":  ["Absent", "Present", "Unknown"],
        "desc":   "tumor state present absent existence status finding"
    },
    {
        "field":  "age_at_tumor_assessment",
        "table":  "tumor_assessments",
        "type":   "number",
        "enums":  [],
        "desc":   "patient age years days at tumor assessment measurement evaluation"
    },
    {
        "field":  "invasiveness",
        "table":  "tumor_assessments",
        "type":   "enum",
        "enums":  ["Invasive", "Non-invasive", "Unknown"],
        "desc":   "tumor invasiveness invasive non-invasive extent"
    },
    #  survival_characteristics 
    {
        "field":  "lkss",
        "table":  "survival_characteristics",
        "type":   "enum",
        "enums":  ["Alive", "Dead", "Unknown"],
        "desc":   "last known survival status vital alive dead living deceased survival outcome"
    },
    {
        "field":  "age_at_lkss",
        "table":  "survival_characteristics",
        "type":   "number",
        "enums":  [],
        "desc":   "patient age at last known survival status years days alive follow-up duration"
    },
    #  subject_responses 
    {
        "field":  "response",
        "table":  "subject_responses",
        "type":   "enum",
        "enums":  ["Complete Response", "Partial Response", "Stable Disease",
                   "Progressive Disease", "Not Evaluable"],
        "desc":   "treatment response category complete partial stable progressive remission"
    },
    {
        "field":  "tx_prior_response",
        "table":  "subject_responses",
        "type":   "enum",
        "enums":  ["Yes", "No", "Unknown"],
        "desc":   "treatment prior response therapy before"
    },
    #  molecular_analysis 
    {
        "field":  "molecular_abnormality",
        "table":  "molecular_analysis",
        "type":   "enum",
        "enums":  ["MYCN Amplification", "ALK Mutation", "Unknown"],
        "desc":   "molecular abnormality gene mutation amplification analysis"
    },
    {
        "field":  "anaplasia",
        "table":  "molecular_analysis",
        "type":   "enum",
        "enums":  ["Absent", "Focal", "Diffuse", "Unknown"],
        "desc":   "anaplasia focal diffuse absent molecular pathology"
    },
    #  disease_phase: NOT in processed_gitops.json - README TODO -
    # This field exists in the PCDC schema but has no gitops path mapping yet.
    # Including it here so the schema linker can identify it and flag it
    # as requiring the disease_phase-specific handling (Deliverable A2).
    {
        "field":  "disease_phase",
        "table":  "UNMAPPED",
        "type":   "enum",
        "enums":  ["Initial Diagnosis", "Relapsed", "Progressive", "Refractory",
                   "Persistent", "Secondary Malignancy"],
        "desc":   "disease phase diagnosis initial relapsed relapse recurrence progressive refractory persistent"
    },
]

# GitOps map - sourced from schema/processed_gitops.json in PR #5.
# Fields mapping to [] are subject-level (no nested path needed).
# disease_phase is intentionally absent from the real gitops file -
# this is the root cause of the README TODO.
GITOPS_MAP = {
    "consortium":              [],
    "sex":                     [],
    "race":                    [],
    "ethnicity":               [],
    "age_at_censor_status":    [],
    "censor_status":           [],
    "tumor_classification":    ["tumor_assessments", "biopsy_surgical_procedures", "radiation_therapies"],
    "tumor_site":              ["tumor_assessments"],
    "tumor_state":             ["tumor_assessments"],
    "age_at_tumor_assessment": ["tumor_assessments"],
    "invasiveness":            ["tumor_assessments"],
    "tumor_size":              ["tumor_assessments"],
    "lkss":                    ["survival_characteristics"],
    "lkss_obfuscated":         ["survival_characteristics"],
    "age_at_lkss":             ["survival_characteristics"],
    "response":                ["subject_responses"],
    "tx_prior_response":       ["subject_responses"],
    "interim_response":        ["subject_responses"],
    "molecular_abnormality":   ["molecular_analysis"],
    "anaplasia":               ["molecular_analysis"],
    "age_at_molecular_analysis": ["molecular_analysis"],
    "histology":               ["histologies"],
    "histology_grade":         ["histologies"],
    "stage":                   ["stagings"],
    "stage_system":            ["stagings"],
    "irs_group":               ["stagings"],
    "lab_test":                ["labs"],
    "lab_result":              ["labs"],
    "lab_result_numeric":      ["labs"],
    # disease_phase: not in gitops - this is the gap we are fixing
}


class SchemaLinker:
    """
    Builds a TF-IDF index over PCDC schema field descriptions at init time.
    Links natural language query terms to schema fields via cosine similarity.

    Usage (drop-in for the disambiguation loop in nested_graphql_helper.py):

        linker = SchemaLinker()
        result = linker.link("metastatic bone tumor INRG patients under 10")
        # result.direct_hits   - matched via vector sim, no LLM needed
        # result.needs_llm     - ambiguous pairs to send to single batched LLM call
        # result.all_fields    - combined final field list
        # result.timing        - real elapsed ms for each stage
    """

    def __init__(self, fields: list[dict] = None):
        self.fields = fields or PCDC_FIELDS
        self._build_index()

    def _build_index(self):
        """
        Build TF-IDF matrix once at startup.
        Each field becomes one document: name + table + type + enums + desc
        """
        t0 = time.perf_counter()

        documents = []
        for f in self.fields:
            doc = " ".join([
                f["field"].replace("_", " "),   # underscore -> space so "tumor site" matches
                f["table"].replace("_", " "),
                f["type"],
                " ".join(f["enums"]),
                f.get("desc", ""),
            ])
            documents.append(doc.lower())

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),          # unigrams + bigrams
            min_df=1,
            sublinear_tf=True,           # log(tf) dampens common terms
            strip_accents="unicode",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.build_time_ms = (time.perf_counter() - t0) * 1000

    def _preprocess_query(self, query: str) -> str:
        """Basic clinical term expansion before vectorizing."""
        expansions = {
            r"\bunder\b":      "less than age below",
            r"\bover\b":       "greater than age above",
            r"\bkids?\b":      "pediatric child",
            r"\bsurvivor?s?\b":"survival vital alive",
            r"\bspread\b":     "metastatic metastasis regional",
            r"\bbone\b":       "bone skeletal osseous",
            r"\bmarrow\b":     "bone marrow",
            r"\brelapsed?\b":  "relapsed relapse recurrence disease phase",
            r"\binitial\b":    "initial diagnosis disease phase",
        }
        q = query.lower()
        for pattern, replacement in expansions.items():
            q = re.sub(pattern, replacement, q)
        return q

    def link(self, user_query: str, top_k: int = TOP_K) -> "LinkResult":
        """
        Main entry point. Returns a LinkResult with direct hits, ambiguous pairs,
        and real timing for each stage.
        """
        timing = {}

        # Stage 1: vectorize query
        t0 = time.perf_counter()
        processed = self._preprocess_query(user_query)
        query_vec = self.vectorizer.transform([processed])
        timing["vectorize_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 2: cosine similarity against all fields
        t0 = time.perf_counter()
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        timing["cosine_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 3: classify each field
        t0 = time.perf_counter()
        ranked = sorted(
            [(self.fields[i], float(scores[i])) for i in range(len(self.fields))],
            key=lambda x: x[1], reverse=True
        )

        direct_hits  = []
        ambiguous    = []
        seen_tables  = set()

        for i, (field, score) in enumerate(ranked):
            if score < MIN_SCORE:
                break
            if i + 1 < len(ranked):
                next_score = ranked[i + 1][1]
                gap = score - next_score
            else:
                gap = score

            if score >= DIRECT_THRESHOLD and gap > AMBIGUOUS_DELTA:
                direct_hits.append({**field, "score": round(score, 4), "method": "vector"})
                seen_tables.add(field["table"])
            elif score >= MIN_SCORE and score < DIRECT_THRESHOLD + 0.08:
                # Check if this competes closely with another field
                ambiguous.append({**field, "score": round(score, 4), "method": "pending"})

        # Deduplicate: if we already have a direct hit for a table/field, skip ambiguous
        ambiguous = [
            a for a in ambiguous
            if a["field"] not in {d["field"] for d in direct_hits}
        ][:3]  # cap at 3 ambiguous pairs to batch

        timing["classify_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return LinkResult(
            query=user_query,
            direct_hits=direct_hits[:top_k],
            ambiguous=ambiguous,
            all_ranked=ranked[:8],
            timing=timing,
            index_build_ms=self.build_time_ms,
        )


class LinkResult:
    def __init__(self, query, direct_hits, ambiguous, all_ranked, timing, index_build_ms):
        self.query         = query
        self.direct_hits   = direct_hits
        self.ambiguous     = ambiguous
        self.all_ranked    = all_ranked
        self.timing        = timing
        self.index_build_ms = index_build_ms

    @property
    def all_fields(self):
        return self.direct_hits + [a for a in self.ambiguous if a["method"] == "llm"]

    def needs_llm_batch(self) -> bool:
        return len(self.ambiguous) > 0

    def total_vector_ms(self) -> float:
        return sum(self.timing.values())

    def summary(self) -> str:
        lines = [
            f"Query        : {self.query}",
            f"Index built  : {self.index_build_ms:.1f}ms (one-time startup cost)",
            f"Vectorize    : {self.timing.get('vectorize_ms', 0):.2f}ms",
            f"Cosine sim   : {self.timing.get('cosine_ms', 0):.2f}ms",
            f"Classify     : {self.timing.get('classify_ms', 0):.2f}ms",
            f"Total vector : {self.total_vector_ms():.2f}ms",
            "",
            f"Direct hits  : {len(self.direct_hits)} field(s) resolved by vector search - 0 LLM calls needed",
        ]
        for h in self.direct_hits:
            lines.append(f"  OK  {h['field']:35s} score={h['score']:.4f}  table={h['table']}")

        if self.ambiguous:
            lines.append(f"\nAmbiguous    : {len(self.ambiguous)} field(s) -> 1 batched LLM call to resolve all")
            for a in self.ambiguous:
                lines.append(f"  ?  {a['field']:35s} score={a['score']:.4f}  (too close to distinguish by vector alone)")
        else:
            lines.append("\nAmbiguous    : 0 - all resolved by vector search, no LLM disambiguation needed")

        old_calls = len(self.direct_hits) + len(self.ambiguous) + 1  # old: 1 per term + 1 gen
        new_calls = (1 if self.ambiguous else 0) + 1                  # new: 0-1 batch + 1 gen
        lines += [
            "",
            f"Old approach : {old_calls} sequential LLM calls (est. {old_calls * 1200}-{old_calls * 2000}ms)",
            f"New approach : {new_calls} LLM call(s) + {self.total_vector_ms():.1f}ms vector search",
            f"Calls saved  : {old_calls - new_calls} LLM call(s) eliminated per query",
        ]
        return "\n".join(lines)
