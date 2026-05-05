---
name: hemophilia-sample-generator
description: Generate N synthetic person-records (N specified by the caller; default 100) for a ~100k-population city over the past 6 months, seeded with 2 Hemophilia A patients (one DIAGNOSED on targeted treatment, one UNDIAGNOSED self-managing with general meds) and 2 of their family members (one paired with each patient). Each record carries 5 aspects: demographics, ecommerce, behavioral_logs, social_media, telehealth_data. Use when the user asks to create sample/demo/test data for Hemophilia A cohort analysis or to regenerate `data/hemophilia_sample_data.json`.
---

# Hemophilia A Synthetic Sample Generator

## Purpose

Generate **N** synthetic data records representing individuals from a small city (~100,000 population) over the past six months, where **N is specified by the caller** (the invoking workflow / Python program). The dataset reflects the three primary symptoms of Hemophilia A — **joint swelling, bruising, and bleeding** — and captures five key aspects of information in a concise JSON format.

The N records include:

- **2 Hemophilia A patients** — one formally diagnosed, fully aware, and receiving targeted treatment; one undiagnosed but exhibiting key symptoms, possibly using general medications to manage them.
- **2 family members** — one paired with each patient (sharing zip code). **Each family member shows a pattern similar to their paired patient** across all five aspects: id=3 mirrors id=1's diagnosed/aware profile, id=4 mirrors id=2's undiagnosed/symptom-focused profile. Family members should also maintain distinct non-health-related behavioral traces.
- **(N − 4) general-population residents** — including sports enthusiasts prone to sports-related injuries (potentially showing similar symptoms) and individuals with other medical conditions who may demonstrate higher engagement with health-related social media or telehealth platforms.

## Cohort labels

- `id=1`: `"hemophilia_a_diagnosed_patient"`
- `id=2`: `"hemophilia_a_undiagnosed_patient"`
- `id=3`: `"hemophilia_a_family_diagnosed"` — paired with id=1; mirrors id=1's pattern
- `id=4`: `"hemophilia_a_family_undiagnosed"` — paired with id=2; mirrors id=2's pattern
- `id=5..N`: `"general"`

## Output contract

- **File path**: `<project_root>/data/hemophilia_sample_data.json` (relative to the user's current working directory; create the `data/` directory if missing).
- **Format**: pretty-printed JSON with 2-space indentation. Top-level value is a JSON array of exactly **N** objects (N = the count specified by the caller).
- **Write tool**: use the Write tool exactly once.

---

## Schema — 5 aspects per record

Each record is a JSON object with these top-level keys: `id`, `cohort_label`, `demographics`, `ecommerce`, `behavioral_logs`, `social_media`, `telehealth_data`.

### 1. Demographics

```
"demographics": {
  "zip_code":               str,  // 30 distinct values from the pool "94101"–"94130"
  "gender":                 str,  // "male" | "female" | "unknown"
  "age_range":              str,  // "0-2" | "3-12" | "13-17" | "18-29" | "30-44" | "45-64" | "65+"
  "race":                   str,  // "white" | "yellow" | "black" | "hispanic" | "other"
  "household_income_range": str   // "very_low" | "low" | "lower_middle" | "upper_middle" | "high"
}
```

- N records drawn from 30 zip codes — collisions expected.
- **Family pairing**: id=1 and id=3 share `zip_code`; id=2 and id=4 share `zip_code`. id=3 has a different `age_range` from id=1; id=4 differs from id=2 (parent/child/sibling realism).
- Across all N records: when N ≥ 7, all 7 `age_range` values represented; all 5 `race` values, all 5 `household_income_range` values, ≥1 record with `gender = "unknown"`.

### 2. Ecommerce

```
"ecommerce": {
  "credit_score":       str,   // FICO band: "poor" | "fair" | "good" | "very good" | "exceptional"
  "pharmacy_purchases": {      // medication_name (snake_case) → integer count over past 6 months
    "medication_a": count,
    ...
  }
}
```

All 5 `credit_score` bands must appear across the N records (when N ≥ 5). Every medication key must come from the canonical list below.

#### Canonical medication list

- **Joint swelling**: `acetaminophen`, `topical_diclofenac_gel`, `cold_compress_pack`, `naproxen`
- **Bruising**: `arnica_gel`, `vitamin_k_supplement`, `bromelain`
- **Bleeding**: `tranexamic_acid`, `aminocaproic_acid`, `factor_viii_concentrate`, `desmopressin_nasal_spray`, `hemostatic_gauze`, `styptic_pencil`
- **Background (other conditions / general)**: `loratadine`, `omeprazole`, `metformin`, `atorvastatin`, `albuterol_inhaler`, `multivitamin`, `ibuprofen`, `cetirizine`, `melatonin`

**Prescription-only**: `factor_viii_concentrate`, `tranexamic_acid`, `aminocaproic_acid`, `desmopressin_nasal_spray`. The undiagnosed patient (id=2) and undiagnosed family member (id=4) MUST NOT have these. All other meds are OTC.

#### Pharmacy purchase patterns

The two Hemophilia A patients and their two family members should exhibit **notably higher** purchase counts for medications related to Hemophilia A management or its primary symptoms. Each family member's purchase pattern should resemble their paired patient's:

- **id=1 (diagnosed) and id=3 (family of diagnosed)**: include `factor_viii_concentrate` and additional bleeding-bucket meds; high counts on Hemophilia-relevant meds across all three symptom buckets.
- **id=2 (undiagnosed) and id=4 (family of undiagnosed)**: OTC-only purchases spanning ≥2 symptom buckets with elevated counts; reflects frequent symptom episodes managed with general meds. No prescription-only Hemophilia meds.

Other individuals may also show notable purchase counts for similar-symptom meds due to unrelated conditions — for example, sports enthusiasts buying `cold_compress_pack` + `arnica_gel` + `ibuprofen`, arthritis patients buying `acetaminophen` + `topical_diclofenac_gel`, allergy sufferers with epistaxis buying `vitamin_k_supplement` + `styptic_pencil`. Pharmacy purchases alone should NOT cleanly separate the Hemophilia cohort.

### 3. Behavioral logs (aggregated)

```
"behavioral_logs": {
  "health_related_online_content_view": { keyword_str → int_count, ... },
  "pharmacy_ads_clicks":                { keyword_str → int_count, ... },
  "wellness_search":                    { keyword_str → int_count, ... }
}
```

- All keyword strings must be **strictly fewer than 8 words**.
- Counts are aggregated totals over the past 6 months (integers ≥ 1).

#### Behavioral patterns

- **id=1 (diagnosed patient)**: elevated counts across all 3 sub-categories for content related to Hemophilia A and its primary symptoms.
- **id=2 (undiagnosed patient)**: also elevated counts, but focused on **symptom-related queries and general health concerns** (e.g. "why do i bruise easily", "frequent nosebleeds adults", "joint swelling unknown cause"). May or may not reach content that explicitly references Hemophilia A.
- **id=3 (family of diagnosed)**: caregiver-themed content referencing Hemophilia A and symptom management.
- **id=4 (family of undiagnosed)**: worried-family symptom queries without naming Hemophilia A.
- **General population**: many records show **similarly high activity** levels driven by different medical conditions (chronic pain, diabetes, allergies, mental health) or general interest in personal health. General records MUST NOT include keywords explicitly mentioning "hemophilia" or "factor viii".

### 4. Social media

```
"social_media": {
  "Platform:theme_or_community": "post/comment text with #hashtags",
  ...
}
```

- Maximum **10** key-value pairs per record (top 10 most recent activities).
- **Keys** encode platform + theme/subreddit/hashtag/community (e.g. `"Reddit:r/rare_diseases"`, `"Twitter:#HemophiliaAwareness"`, `"Facebook:NationalHemophiliaFoundation"`, `"PatientsLikeMe:HemophiliaA"`).
- **Values** are short text snippets (≤ ~30 words) representing posts/comments, with relevant hashtags embedded naturally.
- If a platform/community recurs, disambiguate (e.g. `"Reddit:r/Hemophilia#1"`, `"Reddit:r/Hemophilia#2"`).
- Activity may include non-health-related content for realism.

#### Canonical pools

**Hemophilia-associated** (primarily for id=1 and id=3):
- **Subreddits**: `r/rare_diseases`, `r/ChronicIllness`, `r/GeneticDisorders`, `r/AskDocs`, `r/medical`, `r/medicine`, `r/ChronicPain`, `r/Disability`, `r/HealthAnxiety`, `r/pharmacy`, `r/AskDrugNerds`.
- **Hashtags**: `#Hemophilia`, `#HemophiliaA`, `#HemophiliaB`, `#BleedingDisorder`, `#BleedingDisorders`, `#RareDisease`, `#RareDiseases`, `#ChronicIllness`, `#ChronicDisease`, `#GeneticDisorder`, `#InvisibleIllness`, `#PatientAdvocacy`, `#PatientSupport`, `#HealthAwareness`, `#RareDiseaseAwareness`, `#BleedAware`, `#HemophiliaAwareness`, `#WorldHemophiliaDay`, `#FactorVIII`, `#FactorIX`, `#Prophylaxis`, `#PlasmaDonation`.
- **Communities/orgs**: `NationalHemophiliaFoundation`, `HemophiliaFederationOfAmerica`, `WorldFederationOfHemophilia`, `PatientsLikeMe:HemophiliaA`, `HealthUnlocked:Hemophilia`, `Inspire:Hemophilia`, `Inspire:RareDiseaseSupport`.

**General-health** (primarily for id=2, id=4, and general records):
- **Subreddits**: `r/health`, `r/AskDocs`, `r/medical_advice`, `r/medicine`, `r/HealthAnxiety`, `r/depression`, `r/anxiety`, `r/mentalhealth`, `r/ChronicIllness`, `r/ChronicPain`, `r/autoimmune`, `r/diabetes`, `r/cancer`, `r/fitness`, `r/nutrition`, `r/loseit`, `r/keto`, `r/intermittentfasting`, `r/SkincareAddiction`, `r/psychology`.
- **Hashtags**: `#health`, `#wellness`, `#healthcare`, `#chronicillness`, `#chronicpain`, `#mentalhealth`, `#selfcare`, `#wellbeing`, `#nutrition`, `#fitness`, `#diseaseawareness`, `#patientjourney`, `#healthtips`, `#symptoms`, `#medical`, `#preventivecare`, `#lifestylemedicine`, `#healthyliving`, `#bodyawareness`, `#healthcommunity`.

A few entries (`r/AskDocs`, `r/ChronicIllness`, `r/HealthAnxiety`, `#chronicillness`) appear in both pools — overlap is intentional. Classify by overall thematic mix, not single-entry membership.

#### Social media patterns

- **id=1 (diagnosed patient)**: engagement with subreddits/hashtags closely related to Hemophilia A, alongside general health topics.
- **id=2 (undiagnosed patient)**: higher activity around general health and symptom-related content. Does not explicitly reference Hemophilia A communities (at most 0–1 exploratory entry).
- **id=3 (family of diagnosed)**: Hemophilia A caregiving and family-support themes.
- **id=4 (family of undiagnosed)**: worried-family and general-health concerns; no explicit Hemophilia A references.
- **General population**: many records show similarly high activity levels due to other medical conditions or a general interest in health. May include non-health lifestyle/hobby content. MUST NOT reference Hemophilia A or factor VIII.

### 5. Telehealth data (Teladoc)

```
"telehealth_data": {
  "telehealth_visit_count_in_30_days": int,   // ≥ 0, visits within the past 30 days
  "engagement_score":                  float  // [0.0, 1.0], rounded to 2 decimal places
}
```

`engagement_score` represents **long-term engagement intensity** across the user's full interaction history.

**Important**: elevated values are NOT specific to Hemophilia A. Individuals with other medical conditions, or those who are generally proactive about their health, may exhibit similarly high telehealth usage and engagement. Several general records should have engagement scores that equal or exceed those of the Hemophilia cohort.

---

## General-population noise profiles

The (N − 4) general-population records must include realistic noise so no single feature cleanly separates the Hemophilia cohort. Use these proportions of the general population (round to integers; sub-counts must sum to N − 4):

1. **Sports enthusiasts** (~10%): prone to sports-related injuries showing similar symptoms — elevated `cold_compress_pack`, `arnica_gel`, `ibuprofen`, `naproxen`; active on fitness/sports-injury communities; moderate-to-high telehealth engagement.
2. **Other-condition patients** (~10%): arthritis, diabetes, GERD, allergies, autoimmune, etc. — high behavioral-log engagement on their condition, condition-specific pharmacy purchases, and high telehealth usage.
3. **Health enthusiasts** (~6%): generally proactive — high social-media activity on wellness/fitness/nutrition, elevated telehealth engagement, no specific chronic condition.
4. **Typical residents** (remaining ~74%): low-to-moderate activity across all dimensions.

When (N − 4) ≥ 10, ensure each of profiles 1–3 has at least one representative. For very small N (e.g., N ≤ 10), profiles 1–3 may be omitted in favor of typical residents.

---

## Diversity guidance

The dataset must look genuinely varied across runs:

- Rotate medication selections and counts for id=1..4 each run.
- Vary demographic profiles, zip-code assignments, and family-role age ranges.
- Vary keyword phrasings, subreddit selections, hashtag combinations, and post text. Use real-world canonical pool entries — do not fabricate official-sounding org names.
- Spread engagement scores across [0, 0.95] for general records.

## Determinism

- By default, vary values realistically each run — no fixed seed.
- If the user supplies a seed (e.g. `--seed=42`, "use seed 42"), treat it as the RNG seed so the same seed reproduces the same output. Acknowledge the seed when used.

## Self-check before writing

Before calling Write, verify:

1. Exactly **N** records with unique `id` values 1..N (N is the count specified by the caller).
2. Cohort label counts: 1 diagnosed patient, 1 undiagnosed patient, 1 family-diagnosed, 1 family-undiagnosed, (N − 4) general.
3. id=1 and id=3 share `zip_code`; id=2 and id=4 share `zip_code`.
4. id=1 and id=3 have different `age_range`; id=2 and id=4 have different `age_range`.
5. id=1 includes `factor_viii_concentrate`; id=2 and id=4 have NO prescription-only Hemophilia meds.
6. Every record has all 7 top-level keys: `id`, `cohort_label`, `demographics`, `ecommerce`, `behavioral_logs`, `social_media`, `telehealth_data`.
7. All `behavioral_logs` keyword strings are strictly fewer than 8 words.
8. General records have NO behavioral-log keys or social-media references mentioning "hemophilia" or "factor viii".
9. `social_media` ≤ 10 entries per record; each value ≤ ~30 words.
10. `telehealth_data` on every record: integer visit count ≥ 0; float engagement score in [0.0, 1.0] rounded to 2 decimal places.
11. Sports-enthusiast and other-condition noise records present in the general population, with several matching or exceeding the Hemophilia cohort on at least one feature.
12. All categorical fields use only the allowed enum values.
13. JSON is pretty-printed with 2-space indentation.
