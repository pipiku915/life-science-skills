# Hemophilia A Cohort Analysis

Synthetic-data + unsupervised-clustering pipeline for Hemophilia A cohort detection. Generates a realistic cohort dataset via Claude, runs k-means against it, and ablates JSON perspectives to identify which signals actually drive cluster separation.

## Repository layout

```
hemophilia_workflow.py   LangGraph workflow (entry point — `python hemophilia_workflow.py`)
skills/SKILL.md          Synthetic-data generation specification
output_data/             Generated dataset (gitignored)
model_output/            Plots produced by the workflow (gitignored)
```

## What the skill does

`skills/SKILL.md` is the contract Claude follows to generate the synthetic dataset. It produces **N** records (N is set by the caller; default 100) modeling individuals from a ~100k-population city over the past 6 months. Each record carries 5 perspectives — `demographics`, `ecommerce`, `behavioral_logs`, `social_media`, `telehealth_data` — and the dataset is seeded with:

- **id=1** — diagnosed Hemophilia A patient (formally diagnosed, on targeted treatment incl. Factor VIII)
- **id=2** — undiagnosed Hemophilia A patient (key symptoms, OTC meds only)
- **id=3** — family of id=1 (shares zip code, mirrors the diagnosed/aware pattern)
- **id=4** — family of id=2 (shares zip code, mirrors the undiagnosed/symptom-focused pattern)
- **id=5..N** — general-population residents

The general population is intentionally noisy — sports enthusiasts, other-condition patients (arthritis, diabetes, allergies, etc.), and health enthusiasts — so no single feature cleanly separates the Hemophilia cohort. The reflected primary symptoms are joint swelling, bruising, and bleeding.

## Workflow steps (`hemophilia_workflow.py`)

The script is a 4-node LangGraph pipeline. Run it with `python hemophilia_workflow.py`. Adjust `TOTAL_SAMPLES` and `CHUNK_SIZE` at the top of the file to scale.

### Node 1 — `generate_data`
Loads `output_data/hemophilia_sample_records.json` if it already contains exactly `TOTAL_SAMPLES` records; otherwise calls the Claude API to generate it. The API call is **chunked** (1 seeded-cohort call + ceil((N − 4) / CHUNK_SIZE) general-population calls) because Opus 4.7 caps a single response at ~32K tokens. Each chunk's record ids are force-rewritten to a contiguous range so the concatenation is always `1..N`.

### Node 2 — `cluster_model`
Extracts a 41-feature vector per record (demographics ordinals + pharmacy purchases + behavioral-log aggregates + social-media counts + telehealth metrics), `StandardScaler`-normalizes, then fits `KMeans(k=2)`. The cluster that catches the most seeded ground-truth records is mapped to "hemophilia"; the other becomes "general". Reports silhouette score, cluster sizes, and TP/FP/FN/TN against ground truth.

### Node 3 — `visualize`
Plots a **centroid-distance scatter**: each point's x is its Euclidean distance to the hemophilia centroid, y is its distance to the general centroid (both in the original 41-dim feature space). The decision boundary is the diagonal `y = x` — exact, because k-means assigns by nearest centroid. Saved to `model_output/hemophilia_clustering.png`.

### Node 4 — `ablate_driver`
Ranks the 5 JSON perspectives by their share of the inter-cluster squared distance — i.e., which perspective drives the split — then runs two ablations:

1. Drop the **top-1** perspective's columns, refit k-means, render → `model_output/ablation_main_driver_top_1_removed.png`
2. Drop **top-1 + top-2** perspectives' columns together, refit, render → `model_output/ablation_main_driver_top_1and2_removed.png`

Each ablation reports cluster sizes, hemophilia cluster id, silhouette, TP/FP/FN/TN, precision/recall, per-cohort-member hit/miss (`id=1✓ id=2✗ ...`), and the perspectives that **remain** after the drop. Comparing the two plots shows whether the cohort signal is concentrated in one perspective or distributed across several.

## Setup

```bash
pip install anthropic langgraph python-dotenv scikit-learn matplotlib numpy
echo 'ANTHROPIC_API_KEY=...' > .env
python hemophilia_workflow.py
```

For larger N (≥1000), see the chunk-size guidance at the top of `hemophilia_workflow.py`. Beyond ~250 sequential API calls (N ≳ 5000) you'll want per-chunk checkpointing and bounded async concurrency before clicking go.
