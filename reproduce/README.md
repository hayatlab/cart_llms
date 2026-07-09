# Reproducibility package — precomputed intermediate files

This folder holds **precomputed intermediate files** so the pipeline can be reproduced
without rerunning every upstream step. It accompanies:

> *Cell* (2026). https://doi.org/10.1016/j.cell.2026.06.002 — Hayat Group.
> <!-- TODO: paste the full author list / title / volume / pages -->

> **Note for maintainers:** items marked `TODO` below (full citation, the Zenodo link for
> the large single-cell atlases, and atlas provenance/license) still need to be filled in
> before this is considered final.

---

## 1. What the pipeline does

Candidate genes are scored on **five independent dimensions**, the scores are
**aggregated** into a ranked list, and that ranking is used to prime / evaluate
**LLM-based target nomination** (GPT, Claude, Gemini).

| Dimension | Script (repo root) | Output CSV (here) | Main input |
|---|---|---|---|
| Clinical trials | `clinical_trials_scoring.ipynb` | `clinical_trials/clinical_trial_scores.csv` | ClinicalTrials.gov API v2 (live) |
| Human Protein Atlas | `HumanProteinAtlas.ipynb` | `Protein_Atlas_results/gene_scores_skin.csv` | `proteinatlas_4ef89daa.tsv` |
| scRNA differential expression | `Heatmap.ipynb`, `de_scoring.R` | `heatmap_results/skin_results.csv` | skin single-cell atlas (`.h5ad`), GTEx v10 bulk, TCGA (live) |
| GTEx median expression | `gtex_scoring.R` | `gtex_results/median_expression_results.csv` | GTEx Portal API via `gtexr` (live) |
| Protein expression | `Protein_Expression_Scoring.ipynb`, `protein_exp.py` | `protein_expression/protein_expression_results_with_tissues.csv` | UniProt + ProteomicsDB APIs (live) |

Aggregation: `Score_Aggregation.ipynb` reads the **five CSVs above** and writes the final
ranking (`top_100_genes_by_aggregated_score.csv`) plus per-weight-scheme tables
`gene_scores_*.csv`.

LLM nomination: `gemini_model.py`, `multi_request.py`, `prompt_2_new.py` (and the
`_no_base` / `_zero_shot` / `_bulk` variants) and `llm_prompt1_visual*.ipynb` consume the
aggregated per-scheme `gene_scores_*.csv` as prompt priming and query the LLMs.
`_no_base` drops the priming context; `_zero_shot` uses no worked example.

---

## 2. Contents of this folder

- **Five aggregation-input CSVs** (`clinical_trials/`, `Protein_Atlas_results/`,
  `heatmap_results/`, `gtex_results/`, `protein_expression/`) — the outputs of the five
  scoring dimensions. These regenerate the final ranking **without** rerunning the upstream
  scoring.
- **`proteinatlas_4ef89daa.tsv`** (~35 MB) — raw Human Protein Atlas download used by `HumanProteinAtlas.ipynb`.
- **`bulk_review/gene_reads_v10_skin_sun_exposed_lower_leg.gct.gz`** (~37 MB) — GTEx v10 skin bulk read counts used by `de_scoring.R`.
- **`llm_prompt_prep/` and `llm_no_base_prompt_prep/`** — the six per-weight-scheme
  `gene_scores_*.csv` files fed to the LLM prompts (base and no-base variants). The
  zero-shot run uses no pre-baked prep files.
- **`environment.yml`, `updated_requirements.txt`** — the software environment.

### Data NOT included (too large for GitHub — host externally)

The single-cell atlases are multi-GB and must be deposited on Zenodo/figshare (or obtained
from their original source) and downloaded separately:

| File | Size | Used by | Source |
|---|---|---|---|
| `human_skin_ts_after_batcheffect_corrected.h5ad` | ~4.0 GB | `Heatmap.ipynb` (core) | 3CA-derived, batch-corrected |
| `human_skin_before_batcheffect_correction.h5ad` | ~3.5 GB | `Heatmap.ipynb` | 3CA-derived |
| `skin_all_after_subclustering_spp1.h5ad` | ~6.7 GB | `malignant_skin_atlas_downstream_v2.ipynb` | sub-clustered skin |
| `human_Pancreas_ts_after_batcheffect_corrected.h5ad` | ~1.7 GB | downstream comparison | 3CA-derived |
| `skin_ts_human.h5ad` | ~492 MB | downstream comparison | Tabula Sapiens skin |

Original atlases: the **Curated Cancer Cell Atlas (3CA)** and **Tabula Sapiens**.
<!-- TODO: confirm exact source/license for each atlas; deposit the batch-corrected
derivatives and replace with the DOI below. -->
**Download:** `TODO: <ZENODO_DOI or data-repository URL>`

The `de_scoring.R` step also pulls **TCGA-SKCM** via `TCGAbiolinks::GDCquery` at runtime.

---

## 3. How to reproduce

### Path A — final ranking only (fastest)

Uses the five CSVs shipped here; no raw data or API calls needed.

1. Set up the environment (below) and open `Score_Aggregation.ipynb`.
2. Point the five `pd.read_csv(...)` paths at the files in this `reproduce/` folder
   (they are currently hardcoded to absolute paths).
3. Run all cells → `top_100_genes_by_aggregated_score.csv` is the final ranking.

### Path B — full pipeline from scratch

1. Set up the environment and download the large atlases (§2).
2. Run each scoring script/notebook to regenerate the five CSVs.
3. Run `Score_Aggregation.ipynb` to produce the aggregated ranking and per-scheme `gene_scores_*.csv`.
4. Fill in API keys and run the LLM nomination scripts.

### Environment

```bash
conda env create -f reproduce/environment.yml
conda activate primekg                    # env name defined in environment.yml
# pip-only alternative: pip install -r reproduce/updated_requirements.txt
```

The R scripts self-install their packages via an `install_and_load()` helper — record your
`sessionInfo()` when you run them.

---

## 4. Caveats

- **API keys:** the LLM scripts read keys from constants set to `"FILL"` (with `os.environ`
  fallbacks). Set your own OpenAI / Anthropic / Gemini keys before running — never commit real keys.
- **Hardcoded paths:** notebooks use absolute paths; edit them to your local layout (or make them relative).
- **Live-API drift:** the clinical-trials, GTEx, UniProt/ProteomicsDB, and TCGA steps fetch
  data live, so results can change over time — record the access date and API/package
  versions for exact reproducibility.
