# CMTA Survival Prediction

This module predicts survival risk with CMTA from:

- CLAM/WSI pathomics features, usually a `.pt` feature file.
- A user-uploaded single-patient genomic CSV.
- CMTA checkpoint preprocessing metadata, used to restore training-time genomic
  column order, StandardScaler parameters, omic column groups, and q-bins.

The old built-in patient lookup fallback has been removed. The web/API flow must
pass the uploaded genomic CSV path explicitly.

## Attention Outputs

When `return_attention=True`, CMTA also returns cross-modal attention:

- `p_in_g_att`: for each sampled pathology patch, attention over the six omic
  groups. The predictor counts which omic receives the highest score most often.
- `g_in_p_att`: for each omic group, attention over the sampled pathology
  patches. If CLAM coordinates are supplied, the predictor writes six patch
  square heatmaps, one per omic group.

For frontend CLAM output, coordinates are loaded from the `.pt` file parent's
`../h5_files/{slide_id}.h5` path. If OOM sampling reduces pathology patches to
4096, the same sampled indices are applied to the coordinates before heatmap
rendering.

## Required Files

- `cmta_survival/results/{BLCA,BRCA,LUAD}/model_best_*.pth.tar`
- `checkpoint["preprocessing"]` or a sibling `preprocessing_metadata.json`
- A single-patient genomic CSV uploaded by the user
- A pathomics feature tensor from CLAM

## Direct Usage

```python
import numpy as np

from cmta_survival import format_survival_result, predict_survival

pathomics_features = np.random.randn(500, 1024).astype(np.float32)
gene_csv = "patient_genomic.csv"

result = predict_survival(
    "BRCA",
    pathomics_features,
    genomic_features=gene_csv,
)

print(format_survival_result(result))
```

## CSV Format

The uploaded genomic CSV must contain exactly one patient row. For new CMTA
checkpoints, feature construction is driven by checkpoint metadata:

- `genomic_feature_columns` defines the full training-time genomic column order.
- `scaler.mean` and `scaler.scale` standardize those columns.
- `omic_names` defines the exact six omic input column lists.
- `q_bins` defines the survival-time bin boundaries used for postprocessing.

If preprocessing metadata is absent, the predictor falls back to the legacy
`gene_data/signatures.csv` parser for old checkpoints.
