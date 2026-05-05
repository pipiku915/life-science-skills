#!/usr/bin/env python3
"""
LangGraph workflow for Hemophilia A cohort analysis:
  1. Generate synthetic data via Claude API (using the hemophilia-sample-generator skill)
  2. Unsupervised clustering (KMeans) to discover cohort structure
  3. Visualize clusters as a centroid-distance scatter with ground-truth overlay
  4. Identify the top-1 feature driving cluster separation, ablate it, and re-visualize
"""

import json
import math
import re
from pathlib import Path
from typing import TypedDict

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "output_data" / "hemophilia_sample_records.json"
MODEL_OUTPUT = PROJECT_ROOT / "model_output"
SKILL_PATH = Path.home() / ".claude" / "skills" / "hemophilia-sample-generator" / "SKILL.md"

# Single knob for synthetic-record count. Drives the Claude API request,
# prompt content, max_tokens budget, and downstream assertions.
TOTAL_SAMPLES = 1000

# Max records per Claude API call. Opus 4.7 caps response output at ~32K tokens,
# and a single record runs ~1000–1500 tokens, so a chunk of 15 stays comfortably
# under the limit. Generation issues 1 seeded-cohort call (4 records) plus
# ceil((TOTAL_SAMPLES − 4) / CHUNK_SIZE) general-population calls.
CHUNK_SIZE = 20


# ── State schema ──
class WorkflowState(TypedDict):
    data_path: str
    records: list
    features: list  # np arrays aren't TypedDict-friendly; we store as lists
    labels: list          # ground-truth labels (for evaluation)
    cluster_labels: list  # unsupervised cluster assignments
    silhouette: float
    hemophilia_cluster_id: int
    detection_summary: str
    viz_path: str
    top1_driver_name: str
    top1_driver_score: float
    top1_ablation_silhouette: float
    top1_ablation_viz_path: str
    top2_driver_name: str
    top2_driver_score: float
    top1and2_ablation_silhouette: float
    top1and2_ablation_viz_path: str
    perspective_summary: str
    status: str


# ═══════════════════════════════════════════
# Node 1: Generate synthetic data via Claude
# ═══════════════════════════════════════════
def generate_data(state: WorkflowState) -> WorkflowState:
    """Load existing data if available, otherwise generate via Claude API."""
    print("[Node 1] Synthetic data generation...")

    if DATA_PATH.exists():
        with open(DATA_PATH) as f:
            existing = json.load(f)
        if len(existing) == TOTAL_SAMPLES:
            print(f"  -> Found existing {TOTAL_SAMPLES}-record dataset at {DATA_PATH}, skipping API call.")
            return {**state, "data_path": str(DATA_PATH), "records": existing, "status": "data_loaded"}
        print(f"  -> Existing dataset at {DATA_PATH} has {len(existing)} records but TOTAL_SAMPLES={TOTAL_SAMPLES}. Regenerating...")
    else:
        print("  -> No existing data found. Calling Claude API to generate...")
    try:
        _call_claude_api()
    except Exception as e:
        raise RuntimeError(
            f"Claude API data generation failed: {e}\n"
            f"Ensure ANTHROPIC_API_KEY is valid in .env, or place pre-generated data at:\n"
            f"  {DATA_PATH}"
        ) from e
    with open(DATA_PATH) as f:
        records = json.load(f)
    print(f"  -> Saved {len(records)} records to {DATA_PATH}")
    return {**state, "data_path": str(DATA_PATH), "records": records, "status": "data_generated"}


def _call_claude_api():
    """Generate TOTAL_SAMPLES records via chunked Claude API calls and write to
    DATA_PATH.

    Single-shot generation hits Opus 4.7's ~32K-token output cap once N exceeds
    ~30 records, so the request is split into a 4-record seeded-cohort chunk
    followed by general-population chunks of size ~CHUNK_SIZE. The (N − 4)
    general records are split into chunks of as-equal-as-possible size so the
    last call isn't a near-empty straggler.
    """
    skill_content = SKILL_PATH.read_text()
    # Long timeout — each chunk is a sync call that can take a couple of minutes.
    client = anthropic.Anthropic(timeout=900.0)

    n_general = TOTAL_SAMPLES - 4
    n_general_chunks = max(1, math.ceil(n_general / CHUNK_SIZE)) if n_general > 0 else 0
    print(f"  -> Plan: 1 seeded-cohort call (4 records) + "
          f"{n_general_chunks} general-population call(s) "
          f"({n_general} records, max {CHUNK_SIZE} per call)")

    all_records = []

    # Chunk 0: 4 seeded-cohort records (ids 1..4)
    print("  -> Generating seeded cohort (ids 1..4)...")
    all_records.extend(_generate_chunk(
        client, skill_content, kind="seeded",
        start_id=1, end_id=4, chunk_idx=0,
    ))

    # Chunks 1..k: general population, split into evenly-sized batches
    if n_general > 0:
        base, remainder = divmod(n_general, n_general_chunks)
        sizes = [base + 1 if i < remainder else base for i in range(n_general_chunks)]
        next_id = 5
        for chunk_idx, size in enumerate(sizes, start=1):
            end_id = next_id + size - 1
            print(f"  -> Generating general chunk {chunk_idx}/{n_general_chunks} "
                  f"(ids {next_id}..{end_id}, {size} records)...")
            all_records.extend(_generate_chunk(
                client, skill_content, kind="general",
                start_id=next_id, end_id=end_id, chunk_idx=chunk_idx,
            ))
            next_id = end_id + 1

    assert len(all_records) == TOTAL_SAMPLES, \
        f"Expected {TOTAL_SAMPLES} records, got {len(all_records)}"

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(all_records, f, indent=2)


def _generate_chunk(client, skill_content, *, kind, start_id, end_id, chunk_idx):
    """Issue one Claude API call for a contiguous id range. Returns the parsed
    list of records with ids force-rewritten to start_id..end_id so concatenation
    across chunks always yields a clean 1..TOTAL_SAMPLES sequence.
    """
    n = end_id - start_id + 1

    if kind == "seeded":
        instruction = (
            "Generate ONLY the 4 SEEDED COHORT records described in the skill:\n"
            "  id=1: diagnosed Hemophilia A patient (formally diagnosed, on targeted treatment).\n"
            "  id=2: undiagnosed Hemophilia A patient (key symptoms, general OTC meds only — "
            "NO prescription-only Hemophilia meds).\n"
            "  id=3: family of id=1 (shares zip_code with id=1; different age_range; "
            "mirrors id=1's diagnosed/aware pattern across all 5 aspects).\n"
            "  id=4: family of id=2 (shares zip_code with id=2; different age_range; "
            "mirrors id=2's undiagnosed/symptom-focused pattern; NO prescription-only Hemophilia meds).\n"
            "Use seed 30 for deterministic output.\n"
            "Return ONLY a JSON array of exactly 4 objects."
        )
    else:
        instruction = (
            f"Generate ONLY {n} GENERAL-POPULATION records with sequential ids "
            f"id={start_id} through id={end_id}.\n"
            f"Every record's `cohort_label` MUST be exactly \"general\".\n"
            f"This chunk is one slice of a larger {TOTAL_SAMPLES}-record dataset; the seeded "
            f"cohort (id=1..4) is generated separately and other general chunks are also produced "
            f"separately — vary your demographics, zip codes, and behavior keywords from defaults "
            f"to keep the union diverse.\n"
            f"Within this chunk, include a representative mix from the skill's noise profile: "
            f"sports enthusiasts, other-condition patients (arthritis/diabetes/allergies/etc.), "
            f"health enthusiasts, and typical residents — proportional to chunk size.\n"
            f"NO behavioral_logs keys or social_media references mentioning hemophilia or factor viii.\n"
            f"Use seed {30 + chunk_idx * 100} for deterministic but varied output across chunks.\n"
            f"Follow the skill spec EXACTLY for all 5 aspects.\n"
            f"Return ONLY a JSON array of exactly {n} objects."
        )

    # ~1500 tokens per record + 2K headroom; capped at the model's 32K limit.
    max_tokens = min(32000, max(8000, 1500 * n + 2000))

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": (
                f"You are a synthetic-data generator. Follow the skill specification below EXACTLY.\n\n"
                f"<skill>\n{skill_content}\n</skill>\n\n"
                f"{instruction}\n\n"
                f"Return ONLY the JSON array — no markdown fences, no commentary, no explanation."
            ),
        }],
    )

    raw = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()

    if not raw:
        raise RuntimeError(
            f"Claude API returned empty response for ids {start_id}..{end_id}."
        )

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

    s, e = raw.find("["), raw.rfind("]")
    if s != -1 and e != -1 and e > s:
        raw = raw[s:e + 1]

    try:
        records = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  -> Failed to parse JSON for ids {start_id}..{end_id}. "
              f"First 500 chars of response:\n{raw[:500]}")
        raise RuntimeError(f"Claude API returned invalid JSON: {exc}") from exc

    if not isinstance(records, list) or len(records) != n:
        raise RuntimeError(
            f"Expected a JSON array of {n} records for ids {start_id}..{end_id}, "
            f"got {type(records).__name__} of length "
            f"{len(records) if isinstance(records, list) else 'n/a'}"
        )

    # Force ids to the requested contiguous range so concatenation across chunks
    # gives a clean 1..TOTAL_SAMPLES sequence even if the model used different ids.
    for i, rec in enumerate(records):
        rec["id"] = start_id + i

    return records


# ═══════════════════════════════════════════
# Node 2: Train logistic regression
# ═══════════════════════════════════════════

# All medications from the skill spec
ALL_MEDS = [
    "acetaminophen", "topical_diclofenac_gel", "cold_compress_pack", "naproxen",
    "arnica_gel", "vitamin_k_supplement", "bromelain",
    "tranexamic_acid", "aminocaproic_acid", "factor_viii_concentrate",
    "desmopressin_nasal_spray", "hemostatic_gauze", "styptic_pencil",
    "loratadine", "omeprazole", "metformin", "atorvastatin",
    "albuterol_inhaler", "multivitamin", "ibuprofen", "cetirizine", "melatonin",
]

HEMOPHILIA_KEYWORDS = [
    "hemophilia", "factor viii", "factor_viii", "prophylaxis",
    "bleeding disorder", "bleed", "hemo",
]

# Ordinal / categorical encodings for demographics. Skill enumerates only these
# values, so .get() lookups should always hit; -1 is a defensive sentinel.
AGE_RANGE_ORDINAL = {
    "0-2": 0, "3-12": 1, "13-17": 2, "18-29": 3,
    "30-44": 4, "45-64": 5, "65+": 6,
}
INCOME_ORDINAL = {
    "very_low": 0, "low": 1, "lower_middle": 2,
    "upper_middle": 3, "high": 4,
}
GENDER_ID = {"male": 0, "female": 1, "unknown": 2}
RACE_ID = {"white": 0, "yellow": 1, "black": 2, "hispanic": 3, "other": 4}

# Feature names parallel to extract_features() output. Each name traces back to a
# field/aggregate in the source JSON, so the top-driver report is interpretable.
FEATURE_NAMES = (
    [
        "demographics.age_range",
        "demographics.household_income_range",
        "demographics.gender",
        "demographics.race",
    ]
    + [f"ecommerce.pharmacy_purchases.{m}" for m in ALL_MEDS]
    + [
        "ecommerce.pharmacy_purchases.total_count",
        "ecommerce.pharmacy_purchases.num_symptom_meds",
    ]
    + [
        f"behavioral_logs.{cat}.{stat}"
        for cat in [
            "health_related_online_content_view",
            "pharmacy_ads_clicks",
            "wellness_search",
        ]
        for stat in ["total_count", "num_entries", "hemophilia_keyword_count"]
    ]
    + [
        "social_media.hemophilia_keyword_count",
        "social_media.num_entries",
    ]
    + [
        "telehealth_data.telehealth_visit_count_in_30_days",
        "telehealth_data.engagement_score",
    ]
)

# Each perspective's column span [start, end) in the feature vector. The 5
# perspectives match the 5 top-level keys in the source JSON.
PERSPECTIVE_RANGES = {
    "demographics":    (0, 4),
    "ecommerce":       (4, 28),    # 22 medications + 2 aggregates
    "behavioral_logs": (28, 37),   # 3 sub-dicts × 3 stats
    "social_media":    (37, 39),
    "telehealth_data": (39, 41),
}


def extract_features(record):
    """Extract numerical features from a single record."""
    features = []

    # 0. Demographics (4 features). age_range and household_income_range as
    # ordinals; gender and race as integer category ids.
    demo = record.get("demographics", {})
    features.append(AGE_RANGE_ORDINAL.get(demo.get("age_range", ""), -1))
    features.append(INCOME_ORDINAL.get(demo.get("household_income_range", ""), -1))
    features.append(GENDER_ID.get(demo.get("gender", ""), -1))
    features.append(RACE_ID.get(demo.get("race", ""), -1))

    # 1. Pharmacy purchase counts for each canonical medication (22 features)
    pharmacy = record.get("ecommerce", {}).get("pharmacy_purchases", {})
    for med in ALL_MEDS:
        features.append(pharmacy.get(med, 0))

    # 2. Total pharmacy purchases
    features.append(sum(pharmacy.values()))

    # 3. Count of distinct symptom-bucket meds purchased
    symptom_meds = set(ALL_MEDS[:13])  # joint + bruising + bleeding
    features.append(len(set(pharmacy.keys()) & symptom_meds))

    # 4. Behavioral log aggregates
    blog = record.get("behavioral_logs", {})
    for subdict_name in ["health_related_online_content_view", "pharmacy_ads_clicks", "wellness_search"]:
        subdict = blog.get(subdict_name, {})
        features.append(sum(subdict.values()) if subdict else 0)  # total count
        features.append(len(subdict))  # number of entries

        # Count entries with hemophilia-related keywords
        hemo_count = 0
        for key in subdict:
            if any(kw in key.lower() for kw in HEMOPHILIA_KEYWORDS):
                hemo_count += subdict[key]
        features.append(hemo_count)

    # 5. Social media: count of hemophilia-related entries
    social = record.get("social_media", {})
    hemo_social = 0
    for key, val in social.items():
        text = (key + " " + str(val)).lower()
        if any(kw in text for kw in HEMOPHILIA_KEYWORDS):
            hemo_social += 1
    features.append(hemo_social)
    features.append(len(social))  # total social entries

    # 6. Telehealth
    tele = record.get("telehealth_data", {})
    features.append(tele.get("telehealth_visit_count_in_30_days", 0))
    features.append(tele.get("engagement_score", 0.0))

    return features


def cluster_model(state: WorkflowState) -> WorkflowState:
    """Unsupervised KMeans clustering to discover hemophilia cohort."""
    print("[Node 2] Unsupervised clustering (KMeans)...")

    records = state["records"]

    X = np.array([extract_features(r) for r in records])
    # Ground-truth labels (only for evaluation, NOT used by the model)
    y_true = np.array([0 if r["cohort_label"] == "general" else 1 for r in records])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans with k=2 (hemophilia cohort vs general)
    kmeans = KMeans(n_clusters=2, random_state=21, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    sil_score = float(silhouette_score(X_scaled, cluster_labels))

    # Identify which cluster corresponds to hemophilia
    # The hemophilia cluster is the one where the known hemophilia records (id=1..4) land
    hemo_indices = [i for i, r in enumerate(records) if r["cohort_label"] != "general"]
    cluster_counts = {0: 0, 1: 0}
    for idx in hemo_indices:
        cluster_counts[cluster_labels[idx]] += 1
    hemo_cluster = max(cluster_counts, key=cluster_counts.get)

    # Evaluate: how well does the unsupervised clustering match ground truth?
    # Map cluster_labels to binary prediction aligned with ground truth
    y_pred = np.array([1 if c == hemo_cluster else 0 for c in cluster_labels])
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    cluster_sizes = [int((cluster_labels == i).sum()) for i in range(2)]
    summary = (
        f"Cluster 0: {cluster_sizes[0]} samples | "
        f"Cluster 1: {cluster_sizes[1]} samples\n"
        f"  Hemophilia cluster: Cluster {hemo_cluster}\n"
        f"  Silhouette score: {sil_score:.3f}\n"
        f"  vs Ground Truth — TP={tp} FP={fp} FN={fn} TN={tn}\n"
        f"  Precision={precision:.2%}  Recall={recall:.2%}"
    )

    print(f"  -> Silhouette score: {sil_score:.3f}")
    print(f"  -> Cluster sizes: {cluster_sizes}")
    print(f"  -> Hemophilia cohort mapped to Cluster {hemo_cluster}")
    print(f"  -> vs Ground Truth: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  -> Precision={precision:.2%}  Recall={recall:.2%}")

    return {
        **state,
        "features": X_scaled.tolist(),
        "labels": y_true.tolist(),
        "cluster_labels": cluster_labels.tolist(),
        "silhouette": sil_score,
        "hemophilia_cluster_id": int(hemo_cluster),
        "detection_summary": summary,
        "status": "clustered",
    }


# ═══════════════════════════════════════════
# Node 3: Visualize
# ═══════════════════════════════════════════
def _render_centroid_distance_plot(
    X_scaled, clusters, hemo_cluster, y_true, records, title, viz_path,
):
    """Render a centroid-distance scatter to viz_path.

    Each point is plotted by its Euclidean distance to the two k-means cluster
    centroids in the supplied feature space. Decision boundary is y = x — exact
    because k-means assigns by nearest centroid.
    """
    centroid_hemo = X_scaled[clusters == hemo_cluster].mean(axis=0)
    centroid_general = X_scaled[clusters != hemo_cluster].mean(axis=0)

    dist_hemo = np.linalg.norm(X_scaled - centroid_hemo, axis=1)
    dist_general = np.linalg.norm(X_scaled - centroid_general, axis=1)

    fig, ax = plt.subplots(figsize=(13, 9))

    lo = float(min(dist_hemo.min(), dist_general.min()))
    hi = float(max(dist_hemo.max(), dist_general.max()))
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    diag = np.array([lo, hi])

    ax.fill_between(diag, diag, hi, color="#FFCDD2", alpha=0.30)
    ax.fill_between(diag, lo, diag, color="#BBDEFB", alpha=0.30)
    ax.plot(diag, diag, color="black", linestyle="--", linewidth=2.0,
            label="K-means decision boundary (y = x)")

    non_hemo_mask = y_true == 0
    hemo_mask = y_true == 1
    ax.scatter(dist_hemo[non_hemo_mask], dist_general[non_hemo_mask],
               c="#1976D2", edgecolors="white", s=90, linewidths=1.0,
               label=f"Non-Hemophilia (n={non_hemo_mask.sum()})", zorder=5)
    ax.scatter(dist_hemo[hemo_mask], dist_general[hemo_mask],
               c="#D32F2F", edgecolors="white", s=190, linewidths=1.3,
               marker="D", label=f"Hemophilia A cohort (n={hemo_mask.sum()})", zorder=7)

    ax.text(lo + (hi - lo) * 0.20, lo + (hi - lo) * 0.85,
            "Hemophilia cluster\n(closer to hemo centroid)",
            fontsize=10, fontweight="bold", color="#B71C1C",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#B71C1C", alpha=0.85))
    ax.text(lo + (hi - lo) * 0.80, lo + (hi - lo) * 0.15,
            "General cluster\n(closer to general centroid)",
            fontsize=10, fontweight="bold", color="#0D47A1",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#0D47A1", alpha=0.85))

    hemo_indices = [i for i, r in enumerate(records) if r["cohort_label"] != "general"]
    for idx in hemo_indices:
        r = records[idx]
        role = r["cohort_label"].replace("hemophilia_a_", "").replace("_", " ").title()
        hit = clusters[idx] == hemo_cluster
        status = "✓ in cluster" if hit else "✗ missed"
        ax.annotate(f"id={r['id']} · {role}\n[{status}]",
                    xy=(dist_hemo[idx], dist_general[idx]),
                    xytext=(11, 11), textcoords="offset points",
                    fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor="#FFF59D" if hit else "#FFCDD2",
                              edgecolor="black", alpha=0.92))

    for idx in range(len(records)):
        if clusters[idx] == hemo_cluster and records[idx]["cohort_label"] == "general":
            ax.annotate(f"id={records[idx]['id']} FP",
                        xy=(dist_hemo[idx], dist_general[idx]),
                        xytext=(7, 7), textcoords="offset points",
                        fontsize=6.5, color="#555",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="gray", alpha=0.75))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Distance to Hemophilia centroid  (lower → more Hemophilia-like)",
                  fontsize=11)
    ax.set_ylabel("Distance to General centroid  (lower → more General-like)",
                  fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    fig.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize(state: WorkflowState) -> WorkflowState:
    """K-means visualization as a centroid-distance scatter."""
    print("[Node 3] Creating visualization...")

    X_scaled = np.array(state["features"])
    y_true = np.array(state["labels"])
    clusters = np.array(state["cluster_labels"])
    hemo_cluster = state["hemophilia_cluster_id"]
    records = state["records"]

    title = (
        f"K-means Cohort Detection (k=2) — Centroid-Distance View\n"
        f"Silhouette: {state['silhouette']:.3f}   |   "
        f"Boundary y=x is the exact k-means decision in original feature space"
    )

    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    viz_path = str(MODEL_OUTPUT / "hemophilia_clustering.png")
    _render_centroid_distance_plot(
        X_scaled, clusters, hemo_cluster, y_true, records, title, viz_path,
    )

    print(f"  -> Visualization saved to {viz_path}")
    return {**state, "viz_path": viz_path, "status": "visualized"}


# ═══════════════════════════════════════════
# Node 4: Rank perspectives, ablate top-1 and bottom-1
# ═══════════════════════════════════════════
def _ablation_pass(X_scaled, drop_cols, perspective_name, removed_perspectives,
                   contribution_pct, y_true, records, viz_path):
    """Drop drop_cols from X_scaled, refit k-means, render the centroid-distance
    plot to viz_path. Returns the full clustering result plus the list of
    perspectives that remain after the ablation.
    """
    X_ablated = np.delete(X_scaled, drop_cols, axis=1)
    kmeans = KMeans(n_clusters=2, random_state=21, n_init=10)
    clusters = kmeans.fit_predict(X_ablated)
    sil = float(silhouette_score(X_ablated, clusters))

    hemo_idxs = [i for i, r in enumerate(records) if r["cohort_label"] != "general"]
    counts = {0: 0, 1: 0}
    for idx in hemo_idxs:
        counts[int(clusters[idx])] += 1
    hemo_cluster = int(max(counts, key=counts.get))

    cluster_sizes = [int((clusters == 0).sum()), int((clusters == 1).sum())]

    y_pred = (clusters == hemo_cluster).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Compact per-cohort-member hit/miss summary for console + state output
    cohort_hits_parts = []
    for idx in hemo_idxs:
        rec = records[idx]
        in_hemo = int(clusters[idx]) == hemo_cluster
        cohort_hits_parts.append(f"id={rec['id']}{'✓' if in_hemo else '✗'}")
    cohort_hits = "  ".join(cohort_hits_parts) + f"   ({tp}/{len(hemo_idxs)} recovered)"

    remaining = [p for p in PERSPECTIVE_RANGES if p not in removed_perspectives]
    remaining_label = ", ".join(remaining) if remaining else "(none)"

    title = (
        f"Ablation: K-means after removing perspective  '{perspective_name}'  "
        f"({contribution_pct:.1f}% of inter-cluster variance · "
        f"{len(drop_cols)} features dropped)\n"
        f"Cluster sizes: [C0={cluster_sizes[0]}, C1={cluster_sizes[1]}]   ·   "
        f"Hemophilia → Cluster {hemo_cluster}   ·   "
        f"Silhouette: {sil:.3f}   ·   "
        f"TP={tp} FP={fp} FN={fn} TN={tn}   ·   "
        f"P={precision:.2%} R={recall:.2%}\n"
        f"Remaining perspectives ({len(remaining)}): {remaining_label}"
    )
    _render_centroid_distance_plot(
        X_ablated, clusters, hemo_cluster, y_true, records, title, viz_path,
    )
    return {
        "perspective": perspective_name,
        "n_features_dropped": len(drop_cols),
        "silhouette": sil,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "cluster_sizes": cluster_sizes,
        "hemo_cluster_id": hemo_cluster,
        "cohort_hits": cohort_hits,
        "remaining_perspectives": remaining,
        "viz_path": viz_path,
    }


def ablate_driver(state: WorkflowState) -> WorkflowState:
    """Rank the 5 JSON perspectives by contribution to k-means cluster separation,
    then run two cumulative ablations: drop the top-1 perspective only, then drop
    top-1 and top-2 together. The second pass tests whether the cohort signal
    still survives after stripping the two strongest drivers.

    Per-feature contribution to the squared centroid distance is
    (centroid_hemo[i] - centroid_general[i])^2. Summed across a perspective's
    columns, that gives the perspective's share of the inter-cluster gap.
    """
    print("[Node 4] Perspective ablation: top-1 and top-1+2 cumulative...")

    X_scaled = np.array(state["features"])
    clusters = np.array(state["cluster_labels"])
    hemo_cluster = state["hemophilia_cluster_id"]
    y_true = np.array(state["labels"])
    records = state["records"]

    centroid_hemo = X_scaled[clusters == hemo_cluster].mean(axis=0)
    centroid_general = X_scaled[clusters != hemo_cluster].mean(axis=0)
    diff_sq = (centroid_hemo - centroid_general) ** 2
    total_sq = float(diff_sq.sum())

    contributions = {
        name: float(diff_sq[start:end].sum())
        for name, (start, end) in PERSPECTIVE_RANGES.items()
    }
    sorted_persps = sorted(contributions.items(), key=lambda kv: -kv[1])

    print("  -> Perspective contributions to inter-cluster squared distance:")
    summary_lines = []
    for name, val in sorted_persps:
        pct = 100.0 * val / total_sq if total_sq > 0 else 0.0
        n_feat = PERSPECTIVE_RANGES[name][1] - PERSPECTIVE_RANGES[name][0]
        line = f"{name:<18s}  {pct:5.1f}%   ({n_feat} features)"
        print(f"     {line}")
        summary_lines.append(line)

    top1_name, top1_val = sorted_persps[0]
    top2_name, top2_val = sorted_persps[1]
    top1_pct = 100.0 * top1_val / total_sq if total_sq > 0 else 0.0
    top2_pct = 100.0 * top2_val / total_sq if total_sq > 0 else 0.0
    cumulative_pct = top1_pct + top2_pct

    print(f"  -> Top-1 perspective: {top1_name} ({top1_pct:.1f}%)")
    print(f"  -> Top-2 perspective: {top2_name} ({top2_pct:.1f}%)")
    print(f"  -> Cumulative top-1+2 share of inter-cluster variance: {cumulative_pct:.1f}%")

    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Pass 1: drop only the top-1 perspective
    top1_cols = list(range(*PERSPECTIVE_RANGES[top1_name]))
    top1_path = str(MODEL_OUTPUT / "ablation_main_driver_top_1_removed.png")
    top1_result = _ablation_pass(
        X_scaled, top1_cols, top1_name, [top1_name],
        top1_pct, y_true, records, top1_path,
    )
    print(f"  -> Top-1 ({top1_name}) removed:")
    print(f"     Cluster sizes: [Cluster 0={top1_result['cluster_sizes'][0]}, "
          f"Cluster 1={top1_result['cluster_sizes'][1]}]   |   "
          f"Hemophilia → Cluster {top1_result['hemo_cluster_id']}")
    print(f"     Silhouette: {top1_result['silhouette']:.3f}   |   "
          f"TP={top1_result['tp']} FP={top1_result['fp']} "
          f"FN={top1_result['fn']} TN={top1_result['tn']}   |   "
          f"Precision={top1_result['precision']:.2%}  "
          f"Recall={top1_result['recall']:.2%}")
    print(f"     Cohort hits: {top1_result['cohort_hits']}")
    print(f"     Remaining perspectives ({len(top1_result['remaining_perspectives'])}): "
          f"{', '.join(top1_result['remaining_perspectives'])}")
    print(f"     Saved {top1_path}")

    # Pass 2: drop top-1 AND top-2 perspectives together (cumulative)
    top2_cols = list(range(*PERSPECTIVE_RANGES[top2_name]))
    top1and2_cols = sorted(set(top1_cols) | set(top2_cols))
    top1and2_label = f"{top1_name} + {top2_name}"
    top1and2_path = str(MODEL_OUTPUT / "ablation_main_driver_top_1and2_removed.png")
    top1and2_result = _ablation_pass(
        X_scaled, top1and2_cols, top1and2_label, [top1_name, top2_name],
        cumulative_pct, y_true, records, top1and2_path,
    )
    print(f"  -> Top-1+2 ({top1and2_label}) removed:")
    print(f"     Cluster sizes: [Cluster 0={top1and2_result['cluster_sizes'][0]}, "
          f"Cluster 1={top1and2_result['cluster_sizes'][1]}]   |   "
          f"Hemophilia → Cluster {top1and2_result['hemo_cluster_id']}")
    print(f"     Silhouette: {top1and2_result['silhouette']:.3f}   |   "
          f"TP={top1and2_result['tp']} FP={top1and2_result['fp']} "
          f"FN={top1and2_result['fn']} TN={top1and2_result['tn']}   |   "
          f"Precision={top1and2_result['precision']:.2%}  "
          f"Recall={top1and2_result['recall']:.2%}")
    print(f"     Cohort hits: {top1and2_result['cohort_hits']}")
    print(f"     Remaining perspectives ({len(top1and2_result['remaining_perspectives'])}): "
          f"{', '.join(top1and2_result['remaining_perspectives'])}")
    print(f"     Saved {top1and2_path}")

    return {
        **state,
        "top1_driver_name": top1_name,
        "top1_driver_score": top1_pct,
        "top1_ablation_silhouette": top1_result["silhouette"],
        "top1_ablation_viz_path": top1_path,
        "top2_driver_name": top2_name,
        "top2_driver_score": top2_pct,
        "top1and2_ablation_silhouette": top1and2_result["silhouette"],
        "top1and2_ablation_viz_path": top1and2_path,
        "perspective_summary": "\n".join(summary_lines),
        "status": "complete",
    }


# ═══════════════════════════════════════════
# Build LangGraph workflow
# ═══════════════════════════════════════════
def build_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("generate_data", generate_data)
    graph.add_node("cluster_model", cluster_model)
    graph.add_node("visualize", visualize)
    graph.add_node("ablate_driver", ablate_driver)

    graph.set_entry_point("generate_data")
    graph.add_edge("generate_data", "cluster_model")
    graph.add_edge("cluster_model", "visualize")
    graph.add_edge("visualize", "ablate_driver")
    graph.add_edge("ablate_driver", END)

    return graph.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("Hemophilia A Cohort Analysis — LangGraph Workflow")
    print("=" * 60)
    print()

    workflow = build_workflow()

    initial_state = WorkflowState(
        data_path="",
        records=[],
        features=[],
        labels=[],
        cluster_labels=[],
        silhouette=0.0,
        hemophilia_cluster_id=-1,
        detection_summary="",
        viz_path="",
        top1_driver_name="",
        top1_driver_score=0.0,
        top1_ablation_silhouette=0.0,
        top1_ablation_viz_path="",
        top2_driver_name="",
        top2_driver_score=0.0,
        top1and2_ablation_silhouette=0.0,
        top1and2_ablation_viz_path="",
        perspective_summary="",
        status="initialized",
    )

    result = workflow.invoke(initial_state)

    print()
    print("=" * 60)
    print("Workflow complete!")
    print(f"  Data:                    {result['data_path']}")
    print(f"  Silhouette:              {result['silhouette']:.3f}")
    print(f"  Visualization:           {result['viz_path']}")
    all_persps = list(PERSPECTIVE_RANGES)
    top1_remaining = [p for p in all_persps if p != result["top1_driver_name"]]
    top1and2_remaining = [
        p for p in all_persps
        if p not in (result["top1_driver_name"], result["top2_driver_name"])
    ]

    print(f"  Top-1 perspective:       {result['top1_driver_name']} "
          f"({result['top1_driver_score']:.1f}%)")
    print(f"  Top-1 ablation viz:      {result['top1_ablation_viz_path']}")
    print(f"  Top-1 silhouette:        {result['top1_ablation_silhouette']:.3f}")
    print(f"  Top-1 remaining ({len(top1_remaining)}):     {', '.join(top1_remaining)}")
    print(f"  Top-2 perspective:       {result['top2_driver_name']} "
          f"({result['top2_driver_score']:.1f}%)")
    print(f"  Top-1+2 ablation viz:    {result['top1and2_ablation_viz_path']}")
    print(f"  Top-1+2 silhouette:      {result['top1and2_ablation_silhouette']:.3f}")
    print(f"  Top-1+2 remaining ({len(top1and2_remaining)}):   "
          f"{', '.join(top1and2_remaining)}")
    print()
    print("Perspective ranking:")
    print(result["perspective_summary"])
    print()
    print(result["detection_summary"])
    print("=" * 60)
