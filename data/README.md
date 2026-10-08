# Training data

| File | Columns | Used by |
| --- | --- | --- |
| `raw/mbti_1.csv` | `type`, `posts` | Active Keras classifier and original raw-text experiments |
| `processed/processed_text.csv` | `target`, `text` (plus existing CSV index) | Historical processed-text Keras experiment |

The raw dataset contains 16 MBTI personality labels. The CSV contents are preserved during this reorganization; only their paths and the processed filename spelling change. Keep the source data separate from generated model artifacts in `artifacts/`.

The historical `scripts/preprocess.py` script documents the processed export path. Its export is intentionally commented out; inspect the experiment before choosing to regenerate and replace the existing processed dataset.
